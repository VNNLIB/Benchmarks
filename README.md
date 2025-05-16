# VNNLIB benchmarks

This repository provides a reorganized collection of benchmarks from the VNNCOMP, starting from 2022. The benchmarks are categorized into three distinct architectures: fully connected, convolutional, and residual networks. Each category is managed as a Git submodule, allowing you to selectively download individual categories or all of them together.
In this repository, you can also find:
  * 'extract_compress_files_python.py': A script for extracting .gz files and compressing .onnx and .vnnlib files.
  * 'extract_files_windows.bat: A batch file for extracting .gz files on Windows.
  * 'gen_instance.py: A script for generating a custom instances.csv file based on the provided arguments. This allows you to create a personalized benchmark tailored to your needs, such as focusing on a specific architecture or a particular layer.
  * nns.csv: A CSV file used by gen_instance.py.
  * expected_results.csv: A CSV file that provides a list of expected results for instances specified by {model.onnx, property.vnnlib}. These results are sourced from VNNCOMP results for the years 2022, 2023, and 2024.

## Credits

This repository is the result of the work of [Andrea Valentino Ricotti](https://github.com/andyvale) with the University of Genova

## Download architectures

1. **Cloning the Repository with Submodules**:
   To clone the repository along with all submodules, use the following command:
   ```bash
   git clone --recurse-submodules <repository_url>

2. **Initializing and Updating Submodules After Cloning**:
   If you've already cloned the repository without the --recurse-submodules flag, you can initialize and update the submodules by running :
   ```bash
   git submodule update --init --recursive

3. **Download a Specific Submodule**
  If you only need one of the three architectures, you can download it individually:
  ```bash
  git submodule update --init -- <submodule-path>

## Decompressing Files

To work with the files, you will need to decompress them first. Below are the instructions for decompressing the files on both Linux and Windows systems.

### Both Linux and Windows
A script called 'extract_compress_files_python.py' is provided to help manage file compression and extraction. You can use it with the argument "compress" ("c") to compress all .vnnlib and .onnx files into .gz format. Or you can use it withe the argument "extract" ("e") to extract all .onnx and .vnnlib.
You can specify the architecture you want to process by using the --architectures argument. The available options are "conv", "fc" and "res". If you do not specify an architecture, the script will apply the operation (compression or extraction) to all architectures. 

Extraction:
```bash
py extract_compress_files_python.py -m e 
```
Extraction (only in the convolution folder):
```bash
py extract_compress_files_python.py -m e -a conv
```

Compression:
```bash
py extract_compress_files_python.py -m c
```
Compression (only in the convolution folder):
```bash
py extract_compress_files_python.py -m e -a conv
```

### Linux

To decompress the files in the root directory and all subdirectories, run the following command in this directory:

```bash
gunzip * -r
```

### Windows

For Windows users, unfortunately, we have not found any built-in functionality, so we decided to use 7-Zip to decompress all files. We have provided a batch file to automate the decompression process using 7-Zip. Note that you need to add 7-Zip to your environment variables or modify the .bat file as specified in it. Once you have everything set up, simply execute the 'extract_files_windows.ba't file located in this directory.

### Windows Alternative

You might already have software that allows your Windows PC to run Unix commands (e.g., Git Bash). If so, you can use those tools to decompress the files as you would on a Linux system.

## Filter Instances Script

This repository also includes a script, 'gen_instances.py', which allows you to filter and generate a file of neural network instances based on specified criteria. The script leverages the 'nns.csv' file, which contains information about the neural networks within each directory. For the script to work properly, all submodules are required. If you do not want to download all of them, you can specify which ones you do not intend to use in the 'exarc' parameter. 

**Command-Line Arguments**:
  - The script is configurable via command-line arguments, allowing you to specify:
    - Which architectures, benchmarks, or node types to include or exclude, for benchmarks and nodes exclusion you can specify 'all'('*') (only included benchmarks/nodes will be included).
    - Minimum and maximum parameter counts for the networks.
    - Maximum properties counts for the networks.
    - Output file names and directories for the generated instances.

### Examples

- **Excluding Specific Nodes**:
  - To exclude networks containing certain nodes (e.g., `relu` or `add`) and focus on a specific benchmark (e.g., `mnist`), use the following command:
    ```bash
    python gen_instances.py --exnode "relu,add" --inbench "mnist" --max_par 100000 --outdir "./" --outname "my_inst.csv"
    ```

- **Including Specific Architectures**:
  - To include only networks from the `residual` or `fullyconnected` architectures and exclude certain benchmarks (e.g., `cifar10`, `acasxu_2023`), use:
    ```bash
    python gen_instances.py --inarc "residual,fullyconnected" --exbench "cifar10,acasxu_2023"
    ```

- **Including Specific Nodes**:
  - To include only networks that contain `Gemm` and `Relu` layers, use:
    ```bash
    python gen_instances.py --innode "Gemm,Relu"
    ```

- **Keeping Models with Only Certain Nodes**:
  - To include networks that have **only** `Gemm` and `Relu` layers and exclude all other nodes, use:
    ```bash
    python gen_instances.py --exnode "all" --innode "Gemm,Relu"
    ```
