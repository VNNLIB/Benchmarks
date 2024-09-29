from definitions import *

class logic():
    '''
    Class that contains the logic of the application
    path_to_input_dataset: str - Path to the dataset that contains the benchmarks csv file
    path_to_input_instances: str - Path to the folder that contains the instances.csv (they can be in subfolders)
    path_to_output_instances: str - Path to the folder where the instances.csv will be saved (output of the application)
    '''
    def __init__(self, 
                 path_to_input_dataset = NEURAL_NETWORKS_FILE,
                 path_to_input_instances= BENCHMARKS_VNNCOMP_DIR,
                 path_to_output_instances = os.path.join(BENCHMARKS_VNNCOMP_DIR, 'instances.csv'),
                 help_url = GITHUB_REPO,
                 possible_origins = BENCHMARKS
                 ):
        
        self.possible_origins = possible_origins
        self.help_url = help_url
        self.path_to_dataset = path_to_input_dataset
        self.path_to_input_instances = path_to_input_instances
        self.path_to_output_instances = path_to_output_instances
        self.dataframe = load_nns_dataframe(self.path_to_dataset)
        self.all_nodes = get_nodes_types()

        self.calculated_instances = []
        self.included_nodes = []
        self.excluded_nodes = []
        self.included_architectures = []
        self.incuded_benchmarks = sorted(self.possible_origins)
        self.max_params = -1
        self.min_params = -1
    
    def reset_options(self):
        self.path_to_dataset = NEURAL_NETWORKS_FILE
        self.path_to_input_instances = BENCHMARKS_VNNCOMP_DIR
        self.path_to_output_instances = os.path.join(BENCHMARKS_VNNCOMP_DIR, 'instances.csv')

    def prova(self):
        print(f'Filters are: {self.included_nodes}, {self.excluded_nodes}, {self.included_architectures}, {self.max_params}, {self.min_params}')
        self.reset_filters()
        self.get_filtered_instances(self.included_architectures, self.incuded_benchmarks, self.included_nodes, self.excluded_nodes, self.min_params, self.max_params)
        self.write_output_instances()

    def get_benchmarks_sample(self):
        self.reset_filters()
        self.get_filtered_instances(self.included_architectures, self.incuded_benchmarks, self.included_nodes, self.excluded_nodes, self.min_params, self.max_params)
        dictionaries = []
        for i, row in self.dataframe.iterrows():
            #dictionaries.append({'onnx': row['onnx'].replace('.onnx',''), 'architecture': row['architecture'], 'benchmark': row['benchmark'], 'n_params': row['n_params'], 'node_types': row['node_types']})
            dictionaries.append([row['onnx'].replace('.onnx',''), row['architecture'], row['benchmark'], row['n_params'], ", ".join(row['node_types'])])
        #print(f"Benckmarks sample: {dictionaries}")
        return dictionaries

    def reset_filters(self):
        self.dataframe = load_nns_dataframe(self.path_to_dataset)
    
    def filter_benchmarks(self, benchmarks):
        if(len(benchmarks) == 0):#TODO Remove this workaround
            print(f"Removing ALL benchmarks -> {self.possible_origins}")
            self.dataframe = remove_benchmarks(self.dataframe, BENCHMARKS)
        else:
            self.dataframe = keep_benchmarks(self.dataframe, benchmarks)

    def filter_architectures(self, architectures):
        self.dataframe = keep_architectures(self.dataframe, architectures)
    
    def filter_nodes(self, inc_nodes, exc_nodes):
        if('all' in exc_nodes):
            exc_nodes = sorted(set(self.all_nodes) - set(inc_nodes))
        self.dataframe = remove_nodes(self.dataframe, exc_nodes)
        self.dataframe = keep_nodes(self.dataframe, inc_nodes)
    
    def filter_params(self, min_params, max_params):
        self.dataframe = param_range_filter(self.dataframe, min_params, max_params)
    
    def get_filtered_instances(self, architectures, benchmarks, inc_nodes, exc_nodes, min_params, max_params):
        self.filter_benchmarks(benchmarks)
        self.filter_architectures(architectures)
        self.filter_nodes(inc_nodes, exc_nodes)
        self.filter_params(min_params, max_params)
        self.calculated_instances = get_network_tuples(self.dataframe, self.path_to_output_instances)
    
    def write_output_instances(self):
        with open(self.path_to_output_instances, 'w') as file:
            for instance in self.calculated_instances:
                file.write(instance['onnx'] + ',' + instance['vnnlib'] + ',' + instance['timeout']) if instance['timeout'] else file.write(instance['onnx'] + ',' + instance['vnnlib'])
    
#def get_benchmarks_sample():
#    with open('benchmarks_list.csv', 'r') as file:
#        lines = file.readlines()
#    lines = lines[1:]# Remove the header
#    nodes = [line[line.index('['):line.index(']')+1] for line in lines]
#    architecture = [line.split(',')[0] for line in lines]
#    onnx = [line.split(',')[2] for line in lines]
#    n_params = [line.split(',')[-1] for line in lines]
#    benchmark = [line.split(',')[1] for line in lines]
#    dictionaries = []
#    for i in range(len(onnx)):
#        dictionaries.append({'onnx': onnx[i].replace('.onnx',''), 'architecture': architecture[i], 'benchmark': benchmark[i], 'n_params': n_params[i].replace('\n',''), 'node_types': nodes[i]})
#    return dictionaries
#
def get_nodes_types(): 
    return sorted(set([schema.name.lower() for schema in onnx.defs.get_all_schemas()]))
#
#def get_node():
#    return random.choice(get_nodes_types())