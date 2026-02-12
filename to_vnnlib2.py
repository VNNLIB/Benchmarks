# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy~=2.4.2",
#   "onnx~=1.20.1",
#   "pandas~=3.0.0",
#   "vnnlib~=1.0.1",
# ]
# ///
import argparse
import gzip
import re
import shutil
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import cast, IO, Sequence

import numpy as np
import onnx
import pandas as pd
import vnnlib

type ModelInputOutputInfo = Sequence[tuple[Sequence[int | str], str]]


@dataclass
class Config:
    instances_path: Path | None = None
    root: Path = Path(".")
    output_dir: Path = Path("vnnlib2")
    real_mode: bool = False


def _parse_args(args: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        prog="to_vnnlib2.py",
        description="Convert instances from vnnlib v1.0 to vnnlib v2.0.",
    )
    parser.add_argument(
        "instances_path",
        nargs="?",
        type=Path,
        help="An optional path to the instances csv file to process. If not provided, all instances.csv files under the root directory will be processed.",
    )
    parser.add_argument(
        "--root",
        default=Path("."),
        type=Path,
        help="The root directory in which to look for instances.csv files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=Path("vnnlib2"),
        type=Path,
        help="The directory in which to output converted benchmarks. This directory will be ignored by the instances.csv search if it is a child of the specified root directory.",
    )
    parser.add_argument(
        "-R",
        "--real-mode",
        action="store_true",
        help="Specify the types of input and output variables as 'real' instead of inferring the types from the corresponding ONNX files.",
    )
    parsed_args = parser.parse_args(args, namespace=Config())
    return parsed_args


def collect_instances_paths(
    root: Path, exclude_dirs: Sequence[Path] = ()
) -> list[Path]:
    exclude_dirs = [p.resolve() for p in exclude_dirs]
    paths = list(root.iterdir())
    instances_paths = []
    while paths:
        p = paths.pop(0).resolve()
        if p in exclude_dirs:
            continue
        if p.is_dir():
            paths.extend(p.iterdir())
        if p.name == "instances.csv":
            instances_paths.append(p)
    return sorted(instances_paths)


def get_model_input_output_info(
    model_path: Path,
) -> tuple[ModelInputOutputInfo, ModelInputOutputInfo]:
    if model_path.exists():
        with open(model_path, "rb") as f:
            model_proto = onnx.load(f)
    elif (model_gz_path := (model_path.parent / f"{model_path.name}.gz")).exists():
        with gzip.open(model_gz_path, "r") as f:
            f_io = cast(IO[bytes], f)
            model_proto = onnx.load(f_io)
    else:
        raise RuntimeError(f"model not found: {model_path}")
    # vnnlib v1.0 only supports models with a single input, so we filter out inputs with default values
    initializers = {i.name for i in model_proto.graph.initializer}
    return [
        (
            tuple(
                getattr(d, d.WhichOneof("value")) for d in i.type.tensor_type.shape.dim
            ),
            str(onnx.helper.tensor_dtype_to_np_dtype(i.type.tensor_type.elem_type)),
        )
        for i in model_proto.graph.input
        if i.name not in initializers
    ], [
        (
            tuple(
                getattr(d, d.WhichOneof("value")) for d in o.type.tensor_type.shape.dim
            ),
            str(onnx.helper.tensor_dtype_to_np_dtype(o.type.tensor_type.elem_type)),
        )
        for o in model_proto.graph.output
    ]


def rewrite_variable(
    match: re.Match, input_shape: Sequence[int], output_shape: Sequence[int]
) -> str:
    # define a function to fix variable identifiers to use
    # the new indexing syntax
    varname = match.group(1)
    index = list(map(int, match.group(2).split("_")[1:]))
    if varname == "X" and len(index) != input_shape:
        if len(input_shape) == 0:
            assert index == [0]
            return varname
        elif len(index) == 1:
            index = [d.item() for d in np.unravel_index(index, input_shape)]
        else:
            raise NotImplementedError("unsupported input variable indexing")
    elif varname == "Y" and len(index) != output_shape:
        if len(output_shape) == 0:
            assert index == [0]
            return varname
        elif len(index) == 1:
            index = [d.item() for d in np.unravel_index(index, output_shape)]
        else:
            raise NotImplementedError("unsupported output variable indexing")
    return f"{varname}[{','.join(map(str, index))}]"


def convert_scientific_notation(match: re.Match) -> str:
    # define a function to convert constants in scientific notation,
    # to a floating point format
    return f"{match.group(1)}{Decimal(match.group(2))}"


def _update_chunk(
    chunk: str,
    real_mode: bool,
    input_dtype: str,
    output_dtype: str,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
) -> str:
    # remove const and fun declarations
    chunk = re.compile(r"\(\W*declare-const\W*\w+\W*Real\W*\)\n*").sub("", chunk)
    chunk = re.compile(r"\(\W*declare-fun\W*\w+\W*()\W*Real\W*\)\n*").sub("", chunk)
    if not chunk:
        return chunk
    # rewrite scientific notation
    chunk = re.compile(r"(\W+)([+-]?[0-9]+(\.[0-9]+)?[eE][-+]?[0-9]+)").sub(
        convert_scientific_notation,
        chunk,
    )
    # # rewrite integers to floats
    if real_mode or (
        input_dtype.startswith("float") and output_dtype.startswith("float")
    ):
        chunk = re.compile(r"([^0-9._])([+-]?)(\d+)([^0-9.])").sub(r"\1\2\3.0\4", chunk)
    elif input_dtype.startswith("float"):
        chunk = re.compile(r"(X(_\d+)+\W+)([+-]?)(\d+)([^0-9.])").sub(
            r"\1\3\4.0\5", chunk
        )
        chunk = re.compile(r"([^0-9._])([+-]?)(\d+)(\W+X(_\d+)+)").sub(
            r"\1\2\3.0\4", chunk
        )
    elif output_dtype.startswith("float"):
        chunk = re.compile(r"(Y(_\d+)+\W+)([+-]?)(\d+)([^0-9.])").sub(
            r"\1\3\4.0\5", chunk
        )
        chunk = re.compile(r"([^0-9._])([+-]?)(\d+)(\W+Y(_\d+)+)").sub(
            r"\1\2\3.0\4", chunk
        )
    # rewrite variable names
    chunk = re.compile(r"([XY])(_\d+)+").sub(
        partial(rewrite_variable, input_shape=input_shape, output_shape=output_shape),
        chunk,
    )
    # drop extra newlines that may have appeared
    chunk = re.compile(r"\n\n+").sub("\n\n", chunk)

    return chunk


def update_vnnlib(
    spec_path: Path,
    model_input_output_info: tuple[ModelInputOutputInfo, ModelInputOutputInfo],
    real_mode: bool = False,
):
    # get the input and output shapes and data types
    # for vnnlib v1.0, only a single input and output are supported
    [(_input_shape, input_dtype)], [(_output_shape, output_dtype)] = (
        model_input_output_info
    )
    input_shape = [1 if isinstance(d, str) else d for d in _input_shape]
    output_shape = [1 if isinstance(d, str) else d for d in _output_shape]
    # if we're using real-mode in the updated spec, update the data types
    if real_mode:
        input_dtype = output_dtype = "Real"

    with tempfile.NamedTemporaryFile("w+") as f:
        # read in the original vnnlib v1.0 spec
        if spec_path.exists():
            open_func_r = partial(open, spec_path, mode="r")
            open_func_w = partial(open, f, mode="w")
        elif (spec_gz_path := (spec_path.parent / f"{spec_path.name}.gz")).exists():
            open_func_r = partial(gzip.open, spec_gz_path, mode="rt")
            open_func_w = partial(gzip.open, f.name, mode="wt")
            # open_func_w = partial(open, f.name, mode="w")
            spec_path = spec_gz_path
        else:
            raise RuntimeError(f"spec not found: {spec_path}")

        with open_func_w() as updated_f:
            # add the vnnlib version and network declaration
            updated_f.write(
                "\n".join(
                    [
                        "(vnnlib-version <2.0>)\n",
                        "(declare-network N",
                        f"    (declare-input  X {input_dtype} {input_shape})",
                        f"    (declare-output Y {output_dtype} {output_shape})",
                        ")",
                    ]
                )
            )

            chunk = ""
            with open_func_r() as original_f:
                while new_chunk := original_f.readline():
                    chunk += new_chunk
                    # if the current chunk contains a declaration, wait until we have
                    # balanced parentheses so that the regular expressions in the
                    # update function work as intended
                    if "declare" not in chunk or chunk.count("(") == chunk.count(")"):
                        updated_chunk = _update_chunk(
                            chunk,
                            real_mode=real_mode,
                            input_dtype=input_dtype,
                            output_dtype=output_dtype,
                            input_shape=input_shape,
                            output_shape=output_shape,
                        )
                        updated_f.write(updated_chunk)
                        chunk = ""
            assert len(chunk) == 0
        shutil.copyfile(f.name, spec_path)
        # shutil.copyfile(f.name, "test.vnnlib")

    # check that the updated spec is parseable
    # doesn't work if file is compressed, so skip in that case
    if spec_path.suffix == ".vnnlib":
        _ = vnnlib.parse_query_file(str(spec_path))


def main(args: list[str] | None = None) -> None:
    config = _parse_args(args)

    # get a list of instances.csv files to process
    if config.instances_path is not None:
        instances_paths = [config.instances_path]
    else:
        instances_paths = collect_instances_paths(config.root, [config.output_dir])

    # process all instances.csv files
    for instances_idx, instances_path in enumerate(instances_paths):
        benchmark_dir = instances_path.parent.relative_to(Path.cwd(), walk_up=True)
        shutil.copytree(
            benchmark_dir, config.output_dir / benchmark_dir, dirs_exist_ok=True
        )

        df = pd.read_csv(
            instances_path, header=None, names=("model", "spec", "timeout")
        )

        model_input_output_info: dict[
            Path,
            tuple[ModelInputOutputInfo, ModelInputOutputInfo],
        ] = {}
        converted: dict[
            Path,
            tuple[ModelInputOutputInfo, ModelInputOutputInfo],
        ] = {}
        model_path: Path
        spec_path: Path
        for row_idx, model_path, spec_path, _ in df.itertuples():
            status = (
                "progress"
                f" | benchmarks: {instances_idx} / {len(instances_paths)}"
                f" | instances: {row_idx} / {len(df)}"
                f" | {benchmark_dir}"
            )
            print((status + " " * 100)[:120], end="\r")
            model_path = config.output_dir / benchmark_dir / model_path
            spec_path = config.output_dir / benchmark_dir / spec_path

            if model_path not in model_input_output_info:
                model_input_output_info[model_path] = get_model_input_output_info(
                    model_path
                )

            if spec_path in converted:
                if converted[spec_path] != model_input_output_info[model_path]:
                    raise RuntimeError(
                        f"same spec used with different model interfaces: {spec_path}"
                    )
                continue

            update_vnnlib(
                spec_path, model_input_output_info[model_path], config.real_mode
            )
            converted[spec_path] = model_input_output_info[model_path]
    print()


if __name__ == "__main__":
    main()
