import ast
import os
import random
import pandas as pd
import onnx

GITHUB_REPO = "https://github.com/VNNLIB/Benchmarks"
BENCHMARKS_VNNCOMP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULLYCONNECTED_BENCHMARKS_VNNCOMP_DIR = os.path.join(BENCHMARKS_VNNCOMP_DIR, 'Benchmarks_fc')
CONVOLUTIONAL_BENCHMARKS_VNNCOMP_DIR = os.path.join(BENCHMARKS_VNNCOMP_DIR, 'Benchmarks_conv')
RESIDUAL_BENCHMARKS_VNNCOMP_DIR = os.path.join(BENCHMARKS_VNNCOMP_DIR, 'Benchmarks_residual')
EXPECTED_RESULTS_FILE = os.path.join(BENCHMARKS_VNNCOMP_DIR, 'expected_results.csv')
NEURAL_NETWORKS_FILE = os.path.join(BENCHMARKS_VNNCOMP_DIR, 'nns.csv')
DEBUG_FILE = os.path.join(BENCHMARKS_VNNCOMP_DIR, 'filtered_nns.csv')
ARCHITECTURES = ['convolutional', 'fullyconnected', 'residual']
ONNX_NODES = sorted([schema.name.lower() for schema in onnx.defs.get_all_schemas()]) # Get all the ONNX nodes and convert them to lowercase
BENCHMARKS = sorted(set(net.lower() for path in [FULLYCONNECTED_BENCHMARKS_VNNCOMP_DIR, CONVOLUTIONAL_BENCHMARKS_VNNCOMP_DIR, RESIDUAL_BENCHMARKS_VNNCOMP_DIR]
                        for net in os.listdir(path) if os.path.isdir(os.path.join(path, net))))# Get all the benchmarks and convert them to lowercase
RESULTS = ['sat', 'unsat', 'known', '*']


def check_emptiness (df: pd.DataFrame, lst: list, list_name:str, is_a_removal:bool) -> bool:
    '''Check if the dataframe is empty or the list is empty, a verbose message is printed if so'''
    if df.empty:
        ##verbose_print('\tDataframe is already empty...')
        return True
    if lst is None or len(lst) == 0:
        ##verbose_print(f'\t{list_name} to remove list is empty') if is_a_removal else ##verbose_print(f'\t{list_name} to keep list is empty')
        return True
    return False

def keep_generalized(df: pd.DataFrame, lst: list, col_name: str) -> pd.DataFrame:
    '''Keep only the rows in the DataFrame where the values in col_name are in lst'''
    if check_emptiness(df, lst, col_name.capitalize(), False):
        return df
    ##verbose_print(f"\tKeeping only models with {col_name.lower()} in: ", lst)
    
    if col_name == 'node_types':
        mask = df[col_name].apply(lambda nodes: all(node in [node.lower() for node in nodes] for node in lst))
    else:
        mask = df[col_name].str.lower().isin(lst)
    return df[mask]

def remove_generalized(df: pd.DataFrame, lst: list, col_name: str) -> pd.DataFrame:
    '''Remove the rows in the DataFrame where the values in col_name are in lst'''
    if check_emptiness(df, lst, col_name.capitalize(), True):
        return df
    ##verbose_print(f"\tRemoving models with {col_name.lower()} in: ", lst)
    
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
        ##verbose_print('\tNo n.param bounds needed...')
        return df
    if df.empty:
        ##verbose_print('\tDataframe is empty, skipping parameter filter...')
        return df
    if(min_par > max_par):
        ##verbose_print('\tError in n.parameter filter, min_par > max_par...')
        return df
    if min_par != -1:
        ##verbose_print(f"\tRemoving networks with less than {min_par} parameters...")
        df = df[df['n_params'] >= min_par]
    if max_par != -1:
        #verbose_print(f"\tRemoving networks with more than {max_par} parameters...")
        df = df[df['n_params'] <= max_par]
    return df

def filter_dataframe(df: pd.DataFrame, args: dict) -> pd.DataFrame:
    '''Filter the dataset based on the arguments, it only consider 'inarc', 'exarc', 'inbench', 'exbench', 'innode', 'exnode', 'min_par', 'max_par' '''
    #verbose_print("\n-Filtering on architectures:\n")
    df = keep_architectures(df, args['inarc']) #architecture_filter(df, 'inc', args['inarc'])
    df = remove_architectures(df, args['exarc']) #architecture_filter(df, 'exc', args['exarc'])
    #verbose_print("\n-Filtering on benchmarks:\n")
    df = keep_benchmarks(df, args['inbench']) #benchmode_filter(df, 'inc', args['inbench'])
    df = remove_benchmarks(df, args['exbench']) #benchmode_filter(df, 'exc', args['exbench'])
    #verbose_print("\n-Filtering on nodes:\n")
    df = keep_nodes(df, args['innode']) #nodemode_filter(df, 'inc', args['innode'])
    df = remove_nodes(df, args['exnode']) #nodemode_filter(df, 'exc', args['exnode'])
    #verbose_print("\n-Filtering on parameters number:\n")
    df = param_range_filter(df, args['min_par'], args['max_par'])
    #verbose_print("\n")
    return df

def load_nns_dataframe(file_path: str) -> pd.DataFrame:
    '''Reads the CSV file and converts the 'node_types' column to a list of strings'''
    df = pd.read_csv(file_path)
    df['node_types'] = df['node_types'].apply(ast.literal_eval)
    return df

def get_instances_path(ser: pd.Series) -> str:
    '''Convert the series to "instances.csv" file's path in the associated repository with the parametr, this can be done looking at architecture and benchmark columns'''
    path = BENCHMARKS_VNNCOMP_DIR
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

def get_network_tuples(nets_df: pd.Series, outdir:str) -> list:
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
                fields['rel_path_to_onnx'] = os.path.join(os.path.relpath(BENCHMARKS_VNNCOMP_DIR, start=outdir), benchmark_name, from_bench_to_onnx_path)
                #relative path to get in benchmarks_vnncomp + benchmark + path to vnnlib
                fields['rel_path_to_property.vnnlib'] = os.path.join(os.path.relpath(BENCHMARKS_VNNCOMP_DIR, start=outdir), benchmark_name, from_bench_to_vnnlib_path)
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
    if expected_result == 'known':
        #verbose_print(f"Keeping only instances with known results")
        local_network_tuples = [line for line in network_tuples if not expected_results_df[(expected_results_df['onnx'] == line['onnx']) & (expected_results_df['vnnlib'] == line['vnnlib'])].empty]
    else:
        #verbose_print(f"Keeping instances with expected result {arg_dict['result']}")
        local_network_tuples = [line for line in network_tuples if not expected_results_df[(expected_results_df['onnx'] == line['onnx']) & (expected_results_df['vnnlib'] == line['vnnlib']) & (expected_results_df['result'] == arg_dict['result'])].empty]

    return local_network_tuples

def sample_instances(network_tuples: list, max_properties: int) -> list:
    '''Sample the instances based on the maximum number of properties for each network, it returns a new list'''
    # After filtering the instances, take only the first 'maxprop' instances for each networks: this can be done efficiently because the instances are ordered by the onnx name
    if max_properties == 50:
        #verbose_print(f"Keeping all instances for each network")
        return network_tuples
    
    tmp_network_tuples_list = [] # TODO : reafactor this part
    tmp_instance_list = [network_tuples[0]] if len(network_tuples) > 0 else []
    for i in range(1, len(network_tuples)):
        #if the name of the network is different from the previous one get a random sample
        if network_tuples[i]['onnx'] != network_tuples[i-1]['onnx']:
            #verbose_print(f"Adding a sample of {min(len(tmp_instance_list), max_properties)} instances for network {network_tuples[i-1]['onnx']}")
            tmp_network_tuples_list += random.sample(tmp_instance_list, min(len(tmp_instance_list), max_properties))
            tmp_instance_list = []
        tmp_instance_list.append(network_tuples[i])
    # Add the last network
    if len(tmp_instance_list) > 0:
        tmp_network_tuples_list += random.sample(tmp_instance_list, min(len(tmp_instance_list), max_properties))
        #verbose_print(f"Adding a sample of {min(len(tmp_instance_list), max_properties)} instances for network {network_tuples[i-1]['onnx']}")
    return tmp_network_tuples_list