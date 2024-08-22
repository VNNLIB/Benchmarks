# benchmarks_vnncomp

This repository provides a reorganized collection of benchmarks from VNNCOMP, starting from 2022. The benchmarks are categorized into three distinct architectures: fully connected, convolutional, and residual networks. Each category is managed as a Git submodule, allowing you to selectively download individual categories or all of them together.

## Decompressing Files

To work with the networks and properties files, you will need to decompress them first. You can find a detailed explanation of how to do this in the submodules.

## How to Use

1. **Cloning the Repository with Submodules**:
   To clone the repository along with all submodules, use the following command:
   ```bash
   git clone --recurse-submodules <repository_url>

2. **Cloning the Repository with Submodules**:
   If you've already cloned the repository without the --recurse-submodules flag, you can initialize and update the submodules by running:
   ```bash
   git submodule update --init --recursive

## Included Script

This repository also includes a script, gen_instances.py, which allows you to filter and generate a file of neural network instances based on specified criteria. The script leverages the nns.csv file, which contains information about the neural networks within each directory. For the script to work properly, all submodules are required. If you do not want to download all of them, you can specify which ones you do not intend to use in the 'exarc' parameter. 

**Command-Line Arguments**:
  - The script is configurable via command-line arguments, allowing you to specify:
    - Which architectures, benchmarks, or node types to include or exclude.
    - Minimum and maximum parameter counts for the networks.
    - Maximum properties counts for the networks.
    - Output file names and directories for the generated instances.

### Examples

- **Excluding Certain Nodes**:
  - To exclude networks containing specific nodes (e.g., `relu` or `add`) and focus on a specific benchmark (e.g., `mnist`), you can use:
    ```bash
    python gen_instances.py --exnode "relu,add" --inbench "mnist" --max_par 100000 --outdir "./" --outname "my_inst.csv"
    ```

- **Including Specific Architectures**:
  - To include only networks from the `residual` or `fullyconnected` architectures and exclude certain benchmarks:
    ```bash
    python gen_instances.py --inarc "residual,fullyconnected" --exbench "cifar10,acasxu_2023"
    ```