import argparse
import ast
import os
import random
import pandas as pd
import onnx

# Constants lists TODO(AndyVale): Use the constants in arguments completion
ARCHITECTURES = ['convolutional', 'fullyconnected', 'residual']
ONNX_NODES = sorted([schema.name.lower() for schema in onnx.defs.get_all_schemas()]) # Get all the ONNX nodes and convert them to lowercase
BENCHMARKS = sorted(set(net.lower() for path in ['fullyconnected_benchmarks_vnncomp', 'convolutional_benchmarks_vnncomp', 'residual_benchmarks_vnncomp'] 
                        for net in os.listdir(path) if os.path.isdir(os.path.join(path, net))))# Get all the benchmarks and convert them to lowercase

def init_parser() -> argparse.ArgumentParser:
    '''Initializes the parser with the desired arguments'''
    parser = argparse.ArgumentParser(
        description=
"""Description:
    This script generates a CSV file with lines formatted as 'rel_path_to_onnx,rel_path_to_property.vnnlib,timeout', containing networks that match the specified criteria.
    
    To function properly, the script requires a CSV file named 'nns.csv' with the following columns: 'architecture', 'benchmark', 'onnx', 'node_types', 'n_params'.
    You can substitute 'nns.csv' with any other CSV file that contains the same columns to use your own dataset.
    
    If the 'debug' flag is set, the script will create a file named 'filtered_nns.csv' containing the filtered dataset, allowing you to verify if the filtering is working as expected.
    
    The arguments are case-insensitive, but you must specify the correct names for the architecture, benchmark, or node types.
    The available architecture types are: 'fullyconnected', 'convolutional', and 'residual'.
    Benchmarks correspond to the names of directories within the architecture directories.
    Node types refer to the operator names in the ONNX library (https://onnx.ai/onnx/operators/).
""",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=
"""examples:

    py gen_instances.py --exnode "relu,add" --inbench "mnist" --max_par 100000 --outdir "./" --outname "my_inst.csv"
Generates a file containing networks that do NOT have 'relu' or 'add' nodes, are from the 'mnist' benchmark, have at most 100000 parameters and will be saved in the current directory as 'my_inst.csv'.
    
    py gen_instances.py --exarc "fullyconnected" --innode "conv2d" --min_par 1000 --outdir "../../my_dir"
Generates a file containing networks that are NOT from the 'fullyconnected' architecture, have 'conv2d' nodes, have at least 1000 parameters and will be saved in the directory 'my_dir' that is two levels above the current directory.
    
    py gen_instances.py --inarc "residual, fullyconnected" --exbench "cifar10, acasxu_2023"
Generates a file containing networks that are from the 'residual' or 'fullyconnected' architectures, are NOT from the 'cifar10' or 'acasxu_2023' benchmarks and will be saved in the current directory as 'instances.csv'.
    """
    )
    parser.add_argument('--innode', '-in', type=str, required=False, help="List of nodes that must be included in the dataset")
    parser.add_argument('--inbench', '-ib', type=str, required=False, help="List of benchmarks that must be included in the dataset")
    parser.add_argument('--inarc', '-ia', type=str, required=False, help="List of architectures that must be included in the dataset")
    parser.add_argument('--exnode', '-en', type=str, required=False, help="List of nodes that must be excluded from the dataset. Use 'all' or '*' to exclude all nodes that are not in the 'innode' list")
    parser.add_argument('--exbench', '-eb', type=str, required=False, help="List of benchmarks that must be excluded from the dataset. Use 'all' or '*' to exclude all benchmarks that are not in the 'inbench' list")
    parser.add_argument('--exarc', '-ea', type=str, required=False, help="List of architectures that must be excluded from the dataset")
    parser.add_argument('-max_par', '-Mp', type=int, required=False, default=-1, help="Maximum number of parameters for network")
    parser.add_argument('-min_par', '-mp', type=int, required=False, default=-1, help="Minimum number of parameters for network")
    parser.add_argument('--outdir', '-od', type=str, required=False, default="./", help="Path to the desired output directory, default is the current directory")
    parser.add_argument('--outname', '-on', type=str, required=False, default="instances.csv", help="Name of the output file, default is 'instances.csv'")
    parser.add_argument('--maxprop', '-Mprop', type=int, required=False, default=50, help="Maximum number of properties for each network, default is 50 (maximum)")
    parser.add_argument('--debug', '-d', action='store_true', help="Flag to enable debug mode, a file 'filtered_nns.csv' will be created with the filtered dataset")
    # TODO(AndyVale): 
    # Add maximum (minimum) overall properties for each network (?)
    # Add maximum (minimum) time benchmarking for each network (?)
    return parser

def get_args_as_dict(parser) -> dict:
    '''Parses the arguments and cast them to a proper datatype and then returns the arguments as a dictionary with the argument name as the key'''
    args = parser.parse_args()
    dict_args = vars(args)
    # Compute the absolute path of the output directory and set the output file name in the dictionary
    dict_args['outdir'] = os.path.abspath(dict_args['outdir'])
    dict_args['outname'] = os.path.join(dict_args['outdir'], dict_args['outname'])
    # Split the strings and remove the whitespaces and cast them to lowercase

    if dict_args['innode']:
        dict_args['innode'] = dict_args['innode'].split(",")
        dict_args['innode'] = [dict_args['innode'][i].strip().lower()  for i in range(len(dict_args['innode']))]
    if dict_args['inbench']:
        dict_args['inbench'] = dict_args['inbench'].split(",")
        dict_args['inbench'] = [dict_args['inbench'][i].strip().lower()  for i in range(len(dict_args['inbench']))]
    if dict_args['inarc']:
        dict_args['inarc'] = dict_args['inarc'].split(",")
        dict_args['inarc'] = [dict_args['inarc'][i].strip().lower()  for i in range(len(dict_args['inarc']))]

    if dict_args['exnode']:
        # With 'all' (*) keyword, the script will exclude all the nodes that are not in the 'innode' list
        if(dict_args['exnode'].lower() == 'all' or dict_args['exnode'] == '*'):
            dict_args['exnode'] = list(set(ONNX_NODES) - set(dict_args['innode']))# With 'all' keyword, the script will exclude all the nodes that are not in the 'innode' list
        else:
            dict_args['exnode'] = dict_args['exnode'].split(",")
            dict_args['exnode'] = [dict_args['exnode'][i].strip().lower()  for i in range(len(dict_args['exnode']))]
    if dict_args['exbench']:
        # With 'all' (*) keyword, the script will exclude all the benchmarks that are not in the 'inbench' list
        if(dict_args['exbench'].lower() == 'all' or dict_args['exbench'] == '*'):
            dict_args['exbench'] = list(set(BENCHMARKS) - set(dict_args['inbench']))
        else:
            dict_args['exbench'] = dict_args['exbench'].split(",")
            dict_args['exbench'] = [dict_args['exbench'][i].strip().lower()  for i in range(len(dict_args['exbench']))]
    if dict_args['exarc']:
        dict_args['exarc'] = dict_args['exarc'].split(",")
        dict_args['exarc'] = [dict_args['exarc'][i].strip().lower()  for i in range(len(dict_args['exarc']))]

    # Cast the 'max_par' and 'min_par' is not needed since they are already in the correct datatype

    return dict_args

def check_emptiness (df: pd.DataFrame, lst: list, list_name:str, is_a_removal:bool) -> bool:
    '''Check if the dataframe is empty'''
    if df.empty:
        print('\tDataframe is already empty...')
        return True
    if lst is None or len(lst) == 0:
        print(f'\t{list_name} to remove list is empty') if is_a_removal else print(f'\t{list_name} to remove list is empty')
        return True
    return False

def keep_generalized(df: pd.DataFrame, lst: list, col_name: str) -> pd.DataFrame:
    '''Keep only the rows in the DataFrame where the values in col_name are in lst'''
    if check_emptiness(df, lst, col_name.capitalize(), False):
        return df
    print(f"\tKeeping only models with {col_name.lower()} in: ", lst)
    
    if col_name == 'node_types':
        mask = df[col_name].apply(lambda v: all(node in v for node in lst))
    else:
        mask = df[col_name].isin(lst)
        
    return df[mask]

def remove_generalized(df: pd.DataFrame, lst: list, col_name: str) -> pd.DataFrame:
    '''Remove the rows in the DataFrame where the values in col_name are in lst'''
    if check_emptiness(df, lst, col_name.capitalize(), True):
        return df
    print(f"\tRemoving models with {col_name.lower()} in: ", lst)
    
    if col_name == 'node_types':
        mask = df[col_name].apply(lambda v: not any(node in v for node in lst))
    else:
        mask = ~df[col_name].isin(lst)
        
    return df[mask]

def keep_architectures(df: pd.DataFrame, arc_list: list) -> pd.DataFrame:
    return keep_generalized(df, arc_list, 'architecture')

def remove_architectures(df: pd.DataFrame, arc_list: list) -> pd.DataFrame:
    return remove_generalized(df, arc_list, 'architecture')

def keep_benchmarks(df: pd.DataFrame, bench_list: list) -> pd.DataFrame:
    return keep_generalized(df, bench_list, 'benchmark')

def remove_benchmarks(df: pd.DataFrame, bench_list: list) -> pd.DataFrame:
    return remove_generalized(df, bench_list, 'benchmark')

def keep_nodes(df: pd.DataFrame, node_list: list) -> pd.DataFrame:
    return keep_generalized(df, node_list, 'node_types')

def remove_nodes(df: pd.DataFrame, node_list: list) -> pd.DataFrame:
    return remove_generalized(df, node_list, 'node_types')

def param_filter(df: pd.DataFrame, min_par: int, max_par: int) -> pd.DataFrame:
    '''Filter the dataframe based on the minimum and maximum number of parameters'''
    if(min_par == -1 and max_par == -1):
        print('\tNo n.param bounds needed...')
        return df
    if df.empty:
        print('\tDataframe is empty, skipping parameter filter...')
        return df
    if(min_par > max_par):
        print('\tError in n.parameter filter, min_par > max_par...')
        return df
    if min_par != -1:
        print(f"\tRemoving networks with less than {min_par} parameters...")
        df = df[df['n_params'] >= min_par]
    if max_par != -1:
        print(f"\tRemoving networks with more than {max_par} parameters...")
        df = df[df['n_params'] <= max_par]
    return df

def filter_dataframe(df: pd.DataFrame, args: dict) -> pd.DataFrame:
    '''Filter the dataset based on the arguments'''
    print("\n-Filtering on architectures:\n")
    df = keep_architectures(df, args['inarc']) #architecture_filter(df, 'inc', args['inarc'])
    df = remove_architectures(df, args['exarc']) #architecture_filter(df, 'exc', args['exarc'])
    print("\n-Filtering on benchmarks:\n")
    df = keep_benchmarks(df, args['inbench']) #benchmode_filter(df, 'inc', args['inbench'])
    df = remove_benchmarks(df, args['exbench']) #benchmode_filter(df, 'exc', args['exbench'])
    print("\n-Filtering on nodes:\n")
    df = keep_nodes(df, args['innode']) #nodemode_filter(df, 'inc', args['innode'])
    df = remove_nodes(df, args['exnode']) #nodemode_filter(df, 'exc', args['exnode'])
    print("\n-Filtering on parameters number:\n")
    df = param_filter(df, args['min_par'], args['max_par'])
    print("\n")
    return df

def read_clean_dataframe(file_path: str) -> pd.DataFrame:
    '''Reads the CSV file and cleans it, for example, it lowercases the strings and converts the list of nodes to a list'''
    df = pd.read_csv(file_path)
    df['node_types'] = df['node_types'].str.lower()
    df['architecture'] = df['architecture'].str.lower()
    df['benchmark'] = df['benchmark'].str.lower()
    df['node_types'] = df['node_types'].apply(ast.literal_eval)
    return df

def get_instances_path(ser: pd.Series) -> str:
    '''Convert the series to path in the actual repository, this can be done looking at architecture, benchmark, onnx columns'''
    path = './'
    if ser.empty:
        raise ValueError('Series is empty')

    if(ser['architecture'] is None or ser['benchmark'] is None):
        raise ValueError('One of the columns is empty')
    
    if(ser['architecture'] == 'fullyconnected'):
        path = os.path.join(path, 'fullyconnected_benchmarks_vnncomp')
    elif(ser['architecture'] == 'convolutional'):
        path = os.path.join(path, 'convolutional_benchmarks_vnncomp')
    elif(ser['architecture'] == 'residual'):
        path = os.path.join(path, 'residual_benchmarks_vnncomp')
    else:
        raise ValueError('Architecture is not recognized')
    
    path = os.path.join(path, ser['benchmark'])
    if os.path.exists(path):
        path = os.path.join(path, 'instances.csv')
        if(os.path.isfile(path)):
            return path
        else:
            raise ValueError('Path does not contain instances.csv')
    else:
        raise ValueError('Path does not exist')


if __name__ == '__main__':
    # Parse the arguments and create the dictionary
    parser = init_parser()
    arg_dict = get_args_as_dict(parser)

    # Get the current working directory
    current_working_directory = os.getcwd()
    
    # Read the CSV file and clean it
    nn_df = read_clean_dataframe('nns.csv')
    # Filter the neural networks dataframe based on the dictionary containing the arguments
    filtered_nn_df = filter_dataframe(nn_df, arg_dict)
    # Save the filtered dataframe to a temporary file (to check the output if needed)
    if(arg_dict['debug']):
        filtered_nn_df.to_csv("filtered_nns.csv", index=False)
    # Calculate the output file path and remove it if it already exists
    output_file_path = os.path.join(arg_dict['outdir'], arg_dict['outname'])
    if(os.path.isfile(output_file_path)):
        os.remove(output_file_path)
    # Create the file with the instances that are in the filtered dataframe
    with open(output_file_path, 'w') as output_file:
        for _, net in filtered_nn_df.iterrows():#For each neural network in the filtered dataframe
            # Get the instances file associated with the neural networks
            i_path = get_instances_path(net)
            with open(i_path) as benchmark_instance_file:
                # Get all the properties foreach neural network (matching the onnx name) and shuffle them randomly (to get a random sample of properties)
                network_tuples = [line for line in benchmark_instance_file if net['onnx'] in line]
                random.shuffle(network_tuples)
            # Copy the properties to the output file
            n_of_properties = 0
            for line in network_tuples:#TODO: Add maxPropetiesForEachNetwork to the arguments and
                if n_of_properties >= arg_dict['maxprop']:
                    break
                fields = line.split(',')
                # Change the paths found in the instances file to relative paths from the output directory
                fields[0] = os.path.relpath(os.path.join(os.path.relpath(os.getcwd(), start=arg_dict['outdir']),os.path.dirname(i_path), fields[0]))
                fields[1] = os.path.relpath(os.path.join(os.path.relpath(os.getcwd(), start=arg_dict['outdir']),os.path.dirname(i_path), fields[1]))
                formatted_line = ','.join(fields).replace('\\', '/').replace('\n', '')
                output_file.write(formatted_line + '\n')
                n_of_properties += 1
            if n_of_properties == 0:
                print(f"{n_of_properties} for {net['onnx']} in {i_path}")
    # TODO(AndyVale): Add the autocompletion for the arguments in the parser
    