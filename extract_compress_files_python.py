import os
import gzip
import sys
import shutil
import argparse

def extract_files(folder):
    '''Extracts all .gz files in the specified directory'''
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".gz"):
                with gzip.open(os.path.join(root, file), "rb") as comp_file:
                    with open(os.path.join(root, file).replace(".gz", ""), 'wb') as uncomp_file:
                        shutil.copyfileobj(comp_file, uncomp_file)
                os.remove(os.path.join(root, file))

def compress_files(folder):
    '''Compresses all .vnnlib and .onnx files in the specified directory'''
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".vnnlib") or file.endswith(".onnx"):
                with open(os.path.join(root, file), 'rb') as uncomp_file:
                    with gzip.open(os.path.join(root, file) + ".gz", "wb") as comp_file:
                        shutil.copyfileobj(uncomp_file, comp_file)
                os.remove(os.path.join(root, file))

def main():
    parser = argparse.ArgumentParser(
    prog='extract_compress_files_python',
    description=
"""
------------------------------------------------------------------------------------------------------
This script will extract .gz files and compress .vnnlib and .onnx files.
You can specify the mode (extract or compress) and the architecture (conv, fc, res) to extract/compress.
If no architecture is specified, all architectures will be extracted/compressed.

To extract .gz files, use the extract mode.
To compress .vnnlib and .onnx files, use the compress mode.

examples:
    python extract_compress_files_python.py --mode extract --architectures conv
    python extract_compress_files_python.py --mode compress 

or using the short versions:
    python extract_compress_files_python.py -m e -a conv
    python extract_compress_files_python.py -m c
""",
    epilog=
"""Thank you for using the script! :)
------------------------------------------------------------------------------------------------------""",
    formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('-m', '--mode', choices=['c', 'e'], type=str, help='Mode: c -> compress .onnx and .vnnlib files or e -> extract .gz files', required=True)
    parser.add_argument('-a', '--architectures', type=str, choices=["conv", "fc", "res"], help='Architecture to extract/compress, if not provided all architectures will be extracted/compressed', required=False)
    args = parser.parse_args()
    folder = os.getcwd()
    if not "benchmarks_vnncomp" in folder:
        print("Are you sure you are in the correct folder? You should be in the benchmarks_vnncomp folder...")
    if args.architectures:
        to_folder = dict(conv="convolutional_benchmarks_vnncomp", fc="fullyconnected_benchmarks_vnncomp", res="residual_benchmarks_vnncomp")
        folder = os.path.join(folder, to_folder[args.architectures])
    if args.mode == "compress" or args.mode == "c":
        print(f"Compressing .vnnlib and .onnx files in {folder}")
        compress_files(folder)
    elif args.mode == "extract" or args.mode == "e":
        print(f"Extracting .gz files in {folder}")
        extract_files(folder)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()