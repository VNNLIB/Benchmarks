import argparse
import ast
import os
import random
import pandas as pd
import onnx


def init_parser() -> argparse.ArgumentParser:
    '''Initializes the parser with the desired arguments'''
    parser = argparse.ArgumentParser(
        description=
"""Description:
    This script generates a CSV file with lines formatted as 'rel_path_to_net.onnx,rel_path_to_property.vnnlib,timeout', containing networks that match the specified criteria.
    
    To function properly, the script requires a CSV file named 'nns.csv' with the following columns: 'architecture', 'benchmark', 'onnx', 'node_types', 'n_params'.
    You can substitute 'nns.csv' with any other CSV file that contains the same columns to use your own dataset.
    
    If the 'debug' flag is set, the script will create a file named 'filtered_nns.csv' containing the filtered dataset, allowing you to verify if the filtering based on architecture, benchmark and nodes is working as expected.
    
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
    parser.add_argument('--exnode', '-en', type=str, required=False, help="List of nodes that must be excluded from the dataset. Use 'all' or '*' to exclude all nodes that are not in the 'innode' list")

    parser.add_argument('--inbench', '-ib', type=str, required=False, help="List of benchmarks from which the networks can be picked, default is any benchmark")
    parser.add_argument('--exbench', '-eb', type=str, required=False, help="List of benchmarks from which the networks can not be picked, default is none")

    parser.add_argument('--inarc', '-ia', type=str, required=False, choices=ARCHITECTURES, help="List of architectures from which the networks can be picked, default is any architecture")
    parser.add_argument('--exarc', '-ea', type=str, required=False, choices=ARCHITECTURES, help="List of architectures from which the networks can not be picked, default is none")

    parser.add_argument('--max_par', '-Mp', type=int, required=False, default=-1, help="Maximum number of parameters for network")
    parser.add_argument('--min_par', '-mp', type=int, required=False, default=-1, help="Minimum number of parameters for network")

    parser.add_argument('--maxprop', '-Mprop', type=int, required=False, default=50, help="Maximum number of properties for each network, default is 50 (maximum)")

    parser.add_argument('--result', '-r', type=str, required=False, choices=RESULTS, default='*', help="Filter the instances based on the expected result")

    parser.add_argument('--outdir', '-od', type=str, required=False, default="./", help="Path to the desired output directory, default is the current directory (from where the script is executed)")
    parser.add_argument('--outname', '-on', type=str, required=False, default="instances", help="Name of the output file, default is 'instances', the extension will always be '.csv'")

    parser.add_argument('--debug', '-d', action='store_true', help="Flag to enable debug mode, a file 'filtered_nns.csv' will be created with the filtered dataset")
    parser.add_argument('--verbosity', '-v', action='store_true', help="Flag to enable verbosity mode, the script will print more information")
    # TODO(AndyVale):
    # Add maximum (minimum) time benchmarking for each network (?)
    return parser

def get_args_as_dict(parser) -> dict:
    '''Parses the arguments and cast them to a proper datatype and then returns the arguments as a dictionary with the argument name as the key'''
    args = parser.parse_args()
    dict_args = vars(args)
    
    # Split the strings, remove the whitespaces and cast them to lowercase
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
        dict_args['exbench'] = dict_args['exbench'].split(",")
        dict_args['exbench'] = [dict_args['exbench'][i].strip().lower()  for i in range(len(dict_args['exbench']))]
    if dict_args['exarc']:
        dict_args['exarc'] = dict_args['exarc'].split(",")
        dict_args['exarc'] = [dict_args['exarc'][i].strip().lower()  for i in range(len(dict_args['exarc']))]

    # Cast the 'max_par' and 'min_par' is not needed since they are already in the correct datatype

    return dict_args

def check_emptiness (df: pd.DataFrame, lst: list, list_name:str, is_a_removal:bool) -> bool:
    '''Check if the dataframe is empty or the list is empty, a verbose message is printed if so'''
    if df.empty:
        verbose_print('\tDataframe is already empty...')
        return True
    if lst is None or len(lst) == 0:
        verbose_print(f'\t{list_name} to remove list is empty') if is_a_removal else verbose_print(f'\t{list_name} to keep list is empty')
        return True
    return False

def keep_generalized(df: pd.DataFrame, lst: list, col_name: str) -> pd.DataFrame:
    '''Keep only the rows in the DataFrame where the values in col_name are in lst'''
    if check_emptiness(df, lst, col_name.capitalize(), False):
        return df
    verbose_print(f"\tKeeping only models with {col_name.lower()} in: ", lst)
    
    if col_name == 'node_types':
        mask = df[col_name].apply(lambda nodes: all(node in [node.lower() for node in nodes] for node in lst))
    else:
        mask = df[col_name].str.lower().isin(lst)
    return df[mask]

def remove_generalized(df: pd.DataFrame, lst: list, col_name: str) -> pd.DataFrame:
    '''Remove the rows in the DataFrame where the values in col_name are in lst'''
    if check_emptiness(df, lst, col_name.capitalize(), True):
        return df
    verbose_print(f"\tRemoving models with {col_name.lower()} in: ", lst)
    
    if col_name == 'node_types':
        mask = df[col_name].apply(lambda nodes: not any(node in [node.lower() for node in nodes] for node in lst))
    else:
        mask = ~df[col_name].str.lower().isin(lst)
        
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

def param_range_filter(df: pd.DataFrame, min_par: int, max_par: int) -> pd.DataFrame:
    '''Filter the dataframe based on the minimum and maximum number of parameters'''
    if(min_par == -1 and max_par == -1):
        verbose_print('\tNo n.param bounds needed...')
        return df
    if df.empty:
        verbose_print('\tDataframe is empty, skipping parameter filter...')
        return df
    if(min_par > max_par):
        verbose_print('\tError in n.parameter filter, min_par > max_par...')
        return df
    if min_par != -1:
        verbose_print(f"\tRemoving networks with less than {min_par} parameters...")
        df = df[df['n_params'] >= min_par]
    if max_par != -1:
        verbose_print(f"\tRemoving networks with more than {max_par} parameters...")
        df = df[df['n_params'] <= max_par]
    return df

def filter_dataframe(df: pd.DataFrame, args: dict) -> pd.DataFrame:
    '''Filter the dataset based on the arguments, it only consider 'inarc', 'exarc', 'inbench', 'exbench', 'innode', 'exnode', 'min_par', 'max_par' '''
    verbose_print("\n-Filtering on architectures:\n")
    df = keep_architectures(df, args['inarc']) #architecture_filter(df, 'inc', args['inarc'])
    df = remove_architectures(df, args['exarc']) #architecture_filter(df, 'exc', args['exarc'])
    verbose_print("\n-Filtering on benchmarks:\n")
    df = keep_benchmarks(df, args['inbench']) #benchmode_filter(df, 'inc', args['inbench'])
    df = remove_benchmarks(df, args['exbench']) #benchmode_filter(df, 'exc', args['exbench'])
    verbose_print("\n-Filtering on nodes:\n")
    df = keep_nodes(df, args['innode']) #nodemode_filter(df, 'inc', args['innode'])
    df = remove_nodes(df, args['exnode']) #nodemode_filter(df, 'exc', args['exnode'])
    verbose_print("\n-Filtering on parameters number:\n")
    df = param_range_filter(df, args['min_par'], args['max_par'])
    verbose_print("\n")
    return df

def load_nns_dataframe(file_path: str) -> pd.DataFrame:
    '''Reads the CSV file and converts the 'node_types' column to a list of strings'''
    df = pd.read_csv(file_path)
    df['node_types'] = df['node_types'].apply(ast.literal_eval)
    return df

def get_instances_path(ser: pd.Series) -> str:
    '''Convert the series to "instances.csv" file's path in the associated repository with the parametr, this can be done looking at architecture and benchmark columns'''
    path = ''
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

def get_network_tuples(nets_df: pd.Series, dict_args:dict) -> list:
    '''Get all tuples for the network in the series from the instances.csv files.
    The tuples are a dictionary with the following keys: 'rel_path_to_onnx', 'rel_path_to_property.vnnlib', 'onnx', 'vnnlib', 'timeout' '''
    network_tuples = []
    for _, net in nets_df.iterrows():#For each neural network in the filtered dataframe
        i_path = get_instances_path(net)
        with open(i_path) as benchmark_instance_file:
            # Get all tuples foreach neural network
            lines = benchmark_instance_file.readlines()
        for line in lines:
            if net['onnx'] in line:
                benchmark_name = os.path.dirname(i_path)
                from_bench_to_onnx_path, from_bench_to_vnnlib_path, timeout = line.split(',') if len(line.split(',')) == 3 else line.split(',') + [None]
                fields = {'rel_path_to_onnx': None, 'rel_path_to_property.vnnlib': None, 'onnx': None, 'vnnlib': None, 'timeout': None}
                #relative path to get in benchmarks_vnncomp + benchmark + path to onnx
                fields['rel_path_to_onnx'] = os.path.join(os.path.relpath(BENCHMARKS_VNNCOMP_DIR, start=dict_args['outdir']), benchmark_name, from_bench_to_onnx_path)
                #relative path to get in benchmarks_vnncomp + benchmark + path to vnnlib
                fields['rel_path_to_property.vnnlib'] = os.path.join(os.path.relpath(BENCHMARKS_VNNCOMP_DIR, start=dict_args['outdir']), benchmark_name, from_bench_to_vnnlib_path)
                fields['onnx'] = from_bench_to_onnx_path
                fields['vnnlib'] = from_bench_to_vnnlib_path
                fields['timeout'] = timeout
                network_tuples.append(fields)
    return network_tuples

def filter_instances_by_result(network_tuples: list, expected_result: str) -> list:
    ''''Filter the instances based on the expected result looking at the expected_results.csv file, it returns a new list'''
    if expected_result == '*':
        return network_tuples
    
    expected_results_df = pd.read_csv(EXPECTED_RESULTS_FILE)
    if arg_dict['result'] == 'known':
        verbose_print(f"Keeping only instances with known results")
        local_network_tuples = [line for line in network_tuples if not expected_results_df[(expected_results_df['onnx'] == line['onnx']) & (expected_results_df['vnnlib'] == line['vnnlib'])].empty]
    else:
        verbose_print(f"Keeping instances with expected result {arg_dict['result']}")
        local_network_tuples = [line for line in network_tuples if not expected_results_df[(expected_results_df['onnx'] == line['onnx']) & (expected_results_df['vnnlib'] == line['vnnlib']) & (expected_results_df['result'] == arg_dict['result'])].empty]

    return local_network_tuples

def sample_instances(network_tuples: list, max_properties: int) -> list:
    '''Sample the instances based on the maximum number of properties for each network, it returns a new list'''
    # After filtering the instances, take only the first 'maxprop' instances for each networks: this can be done efficiently because the instances are ordered by the onnx name
    if max_properties == 50:
        verbose_print(f"Keeping all instances for each network")
        return network_tuples
    
    tmp_network_tuples_list = [] # TODO : reafactor this part
    tmp_instance_list = [network_tuples[0]] if len(network_tuples) > 0 else []
    for i in range(1, len(network_tuples)):
        #if the name of the network is different from the previous one get a random sample
        if network_tuples[i]['onnx'] != network_tuples[i-1]['onnx']:
            verbose_print(f"Adding a sample of {min(len(tmp_instance_list), max_properties)} instances for network {network_tuples[i-1]['onnx']}")
            tmp_network_tuples_list += random.sample(tmp_instance_list, min(len(tmp_instance_list), max_properties))
            tmp_instance_list = []
        tmp_instance_list.append(network_tuples[i])
    # Add the last network
    if len(tmp_instance_list) > 0:
        tmp_network_tuples_list += random.sample(tmp_instance_list, min(len(tmp_instance_list), max_properties))
        verbose_print(f"Adding a sample of {min(len(tmp_instance_list), max_properties)} instances for network {network_tuples[i-1]['onnx']}")
    return tmp_network_tuples_list

def write_output_file(dict_args: dict, network_tuples: list) -> None:
    '''Writes the list of network tuples to the output file in the format 'rel_path_to_onnx,rel_path_to_property.vnnlib,timeout(not mandatory)' '''
    # Remove the output file if it already exists
    output_file_path = os.path.join(dict_args['outdir'], dict_args['outname'])
    if(os.path.isfile(output_file_path)):
        verbose_print(f"Removing the existing output file {output_file_path}")
        os.remove(output_file_path)

    # if the outdir does not exist, create it
    if not os.path.exists(dict_args['outdir']):
        verbose_print(f"Creating the output directory {dict_args['outdir']}")
        os.makedirs(dict_args['outdir'])
    
    # Finally write the filtered instances to the output file
    with open(output_file_path, 'w') as output_file:
        for net_tuple in network_tuples:
            row = [net_tuple['rel_path_to_onnx'], net_tuple['rel_path_to_property.vnnlib'], net_tuple['timeout']] if net_tuple['timeout'] else [net_tuple['rel_path_to_onnx'], net_tuple['rel_path_to_property.vnnlib']]
            csv_line = ','.join(row).replace('\\', '/').replace('\n', '')
            output_file.write(csv_line + '\n')

if __name__ == '__main__':
    # Define a bunch of constants
    BENCHMARKS_VNNCOMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    FULLYCONNECTED_BENCHMARKS_VNNCOMP_DIR = os.path.join(BENCHMARKS_VNNCOMP_DIR, 'fullyconnected_benchmarks_vnncomp')
    CONVOLUTIONAL_BENCHMARKS_VNNCOMP_DIR = os.path.join(BENCHMARKS_VNNCOMP_DIR, 'convolutional_benchmarks_vnncomp')
    RESIDUAL_BENCHMARKS_VNNCOMP_DIR = os.path.join(BENCHMARKS_VNNCOMP_DIR, 'residual_benchmarks_vnncomp')
    EXPECTED_RESULTS_FILE = os.path.join(BENCHMARKS_VNNCOMP_DIR, 'expected_results.csv')
    NEURAL_NETWORKS_FILE = os.path.join(BENCHMARKS_VNNCOMP_DIR, 'nns.csv')
    DEBUG_FILE = os.path.join(BENCHMARKS_VNNCOMP_DIR, 'filtered_nns.csv')

    # Check required dir and files are present 
    if not os.path.isfile(NEURAL_NETWORKS_FILE):
        print(f"This script requires the file {NEURAL_NETWORKS_FILE} to be in the same directory as the script")
        exit(1)

    if not os.path.isfile(EXPECTED_RESULTS_FILE):
        print(f"This script requires the file {EXPECTED_RESULTS_FILE} to be in the same directory as the script")
        exit(1)

    if not os.path.isdir(FULLYCONNECTED_BENCHMARKS_VNNCOMP_DIR) or not os.path.isdir(CONVOLUTIONAL_BENCHMARKS_VNNCOMP_DIR) or not os.path.isdir(RESIDUAL_BENCHMARKS_VNNCOMP_DIR):
        print(f"This script requires the directories 'fullyconnected_benchmarks_vnncomp', 'convolutional_benchmarks_vnncomp' and 'residual_benchmarks_vnncomp' to be in the same directory as the script")
        exit(1)

    # Define some lists 
    ARCHITECTURES = ['convolutional', 'fullyconnected', 'residual']
    ONNX_NODES = sorted([schema.name.lower() for schema in onnx.defs.get_all_schemas()]) # Get all the ONNX nodes and convert them to lowercase
    BENCHMARKS = sorted(set(net.lower() for path in [FULLYCONNECTED_BENCHMARKS_VNNCOMP_DIR, CONVOLUTIONAL_BENCHMARKS_VNNCOMP_DIR, RESIDUAL_BENCHMARKS_VNNCOMP_DIR]
                            for net in os.listdir(path) if os.path.isdir(os.path.join(path, net))))# Get all the benchmarks and convert them to lowercase
    RESULTS = ['sat', 'unsat', 'known', '*']

    # Parser setup
    parser = init_parser()
    arg_dict = get_args_as_dict(parser)
    arg_dict['outdir'] = os.path.abspath(arg_dict['outdir'])
    print(f"Output directory: {arg_dict['outdir']}")
    # Remove extension
    arg_dict['outname'] = os.path.splitext(arg_dict['outname'])[0]
    arg_dict['outname'] = arg_dict['outname'].replace('/','').replace('\\','').replace('.','').replace('..','').replace(':','').replace(' ','_').replace('<','').replace('>','').replace('"','').replace('|','').replace('?','').replace('*','').strip()
    arg_dict['outname'] = arg_dict['outname'] + '.csv'
    print(f"Output file name: {arg_dict['outname']}")
    os.chdir(os.path.dirname(__file__))
    if arg_dict['verbosity']:
        def verbose_print(*args):
            print(*args)
    else:
        verbose_print = lambda *a: None

    # Initialize the neural networks dataframe
    nns_df = load_nns_dataframe(NEURAL_NETWORKS_FILE)

    # Filter the neural networks dataframe based on the dictionary containing the arguments
    filtered_nn_df = filter_dataframe(nns_df, arg_dict)
    if(arg_dict['debug']):
        verbose_print(f"Writing the filtered dataset to {DEBUG_FILE}")
        filtered_nn_df.to_csv(DEBUG_FILE, index=False)

    # Get the instances for the filtered networks
    network_tuples = get_network_tuples(filtered_nn_df, arg_dict) 
    if len(network_tuples) == 0:
        print(f"No instances found for given filters")
        exit(0)
    print(f"A total of {len(network_tuples)} instances were found for the given filters")

    # Filter them based on the expected result
    network_tuples = filter_instances_by_result(network_tuples, arg_dict['result'])
    if len(network_tuples) == 0:
        print(f"No instances found for given expected result")
        exit(0)
    if(arg_dict['result'] != '*'):
        print(f"A total of {len(network_tuples)} instances were found for the given expected result")

    # Filter the instances based on the maximum number of properties
    network_tuples = sample_instances(network_tuples, arg_dict['maxprop'])
    if(arg_dict['maxprop'] != 50):
        print(f"A total of {len(network_tuples)} instances were found limited by {arg_dict['maxprop']} number of properties for each network")

    # Write the output file
    write_output_file(arg_dict, network_tuples)