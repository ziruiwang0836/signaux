#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:56:13 2023

@author: wst
"""

import os
from pathlib import Path
import shutil
import json
import copy
import csv
import numpy as np

from neo import SpikeTrain
from elephant.spike_train_generation import StationaryLogNormalProcess as SLNP
from elephant import statistics as ele_stats

from quantities import s, Hz
from .npencoder import NpEncoder

class Signaux(object):
    
    """creates and manipulates synthetic spike trains"""
    
    def __init__(self, network_path, file_name = 'input_config.json', data_path = None, ray_parallel = False, rc = None, role = None, 
                 input_defs = None, inputs = None, neuron_class = None, postsynaptic = None, seed = 23, merge_okay = True):
       
        """
        network_path (str): path to directory containing the network
        file_name (str, optional): file name to write config to. Defaults to 'input_config.json'
        data_path (str, optional): path to directory where data is located, defaults to current working directory 
        ray_parallel (bool, optional): whether to attempt parallel exectuion with ray
        rc (ipyparallel.Client, optional): ipyparallel client, for parallel execution with ipyparallel
        role (str, optional): master or worker, for ipyparallel
        input_defs: (list, optional): list of defined inputs to include, uses predefined parameters
        inputs (list, optional): list of inputs to include, necessary for parallel execution.
                                 If passed to master, inputs must contain complete information (ie., all parameters specified).
        neuron_class(str, optional): specified neuron_class to create input for
        postsynaptic (list, optional): specified neuron ids to create input for
        seed (int, optional): seed for random generator
        merge_okay (bool, optional): whether or not to merge with existing input file
        
        """
        
        self.network_path = network_path
        self.file_name = os.path.join(network_path, file_name)
        
        if data_path:
            self.data_path = data_path
        else:
            self.data_path = os.getcwd()
        
        self.all_parameters = ['generator', 'type', 'frequency', 'start', 'end', 'correlation', 
                              'jitter', 'conductance', 'mod_file', 'parameter_file', 'num_inputs']
        
        self.sigma_map = {'proto':0.7, 'STN': 0.7, 'SNr': 0.6}
        
        self.default_start = 0
        self.default_end = 5


        if inputs ==  None:
            self.use_default_inputs()
            if input_defs ==  None:
                self.inputs = self.defaults
            else:
                assert isinstance(input_defs, list), 'inputs not a list'
                self.inputs = {}
                self.set_up_inputs(input_defs)
        else:
            self.inputs = inputs
            
        if neuron_class ==  None:
            ##revert to default
            self.neuron_class = "SNrn"
        else:
            self.neuron_class = neuron_class
            
        if postsynaptic:
            assert isinstance(postsynaptic, list), 'postsynaptic IDs must be a list'
            self.postsynaptic = postsynaptic
        else:
            #if no postsynaptic neurons are specified create input for just one single neuron
            self.postsynaptic = [0]
            
        self.total_trains = {}
        
        for n in self.postsynaptic:
            self.total_trains[str(n)] = {}
        
        self.signals = []
            
        self.merge_okay = merge_okay
        self.rng = np.random.default_rng(seed = seed)

        self.workers_initialised = False
        self.rc = rc
        self.ray_parallel = ray_parallel
        
        if not role:
            self.role = "master"
        else:
            self.role = role

        assert self.role in ["master", "worker"], \
            "Role must be master or worker."

        if self.role ==  'master' and self.rc is not None:
            self.initialize_parallel()
            
    def initialize_parallel(self):
        """initialize the ipyparallel environment"""
        if self.rc is not None:

            self.d_view = self.rc[:]

            with self.d_view.sync_imports():
                import os
                from signaux import Signaux
                
            self.d_view.push({
                        "network_path": self.network_path,
                        "file_name": self.file_name,
                        "postsynaptic": self.postsynaptic,
                        "inputs": self.inputs, 
                         },
                        block = True)

            cmd_str = ("sg = Signaux(network_path = network_path, role = 'worker', file_name = file_name, postsynaptic = postsynaptic, inputs = inputs)")
            self.d_view.execute(cmd_str, block = True)
            self.workers_initialised = True
            
            print(f"Initialized {len(self.rc)} parallel engines")
        else:
            print("Warning: Parallel mode enabled but no ipyparallel client provided")
            
    def set_up_inputs(self, inputs):
        
        for i in inputs:              
            if isinstance(i, dict):
                for j in list(i.keys()):
                    print(f'Adding {i} input with custom parameters.')
                    if j in list(self.defaults.keys()):
                        if 'n_presynaptic' in i[j].keys():
                            if isinstance(i[j]['n_presynaptic'], list):
                                n_presynaptic = np.random.normal(loc = i[j]['n_presynaptic'][0], scale = i[j]['n_presynaptic'][1])
                            else:
                                n_presynaptic = i[j]['n_presynaptic']
                            for k in range(0, int(n_presynaptic)):
                                self.inputs[j + '_' + str(k)] = self.defaults[j].copy()
                                self.inputs[j + '_' + str(k)].update(i[j])
                        else:
                            self.inputs[j] = self.defaults[j].copy()
                            self.inputs[j].update(i[j])
                    elif j.split('_')[0] in self.defaults:
                        if 'n_presynaptic' in i[j].keys():
                            if isinstance(i[j]['n_presynaptic'], list):
                                n_presynaptic = np.random.normal(loc = i[j]['n_presynaptic'][0], scale = i[j]['n_presynaptic'][1])
                            else:
                                n_presynaptic = i[j]['n_presynaptic']
                            for k in range(0, int(n_presynaptic)):
                                    self.inputs[j + '_' + str(k)] = self.defaults[j.split('_')[0]].copy()
                                    self.inputs[j + '_' + str(k)].update(i[j])
                        else:
                            self.inputs[j] = self.defaults[j.split('_')[0]].copy()
                            self.inputs[j].update(i[j])
                            
                    else:
                        assert all(param in list(i[j].keys()) for param in self.all_parameters), "Incomplete parameter set for input {input_name}.".format(input_name = str(j))
                        self.inputs[j] = i[j]             
                        
            elif isinstance(i, str):
                assert i in list(self.defaults.keys()), 'Unknown input type, please provide complete specifications.'
                print(f'Adding {i} input with default parameters.')
                self.inputs[i] = self.defaults[i].copy()
                
        return
                        
    def csv_inputs(self):
        """Checks for existing csv files or generates them if needed"""
        
        for i in self.inputs:
            print(f'Preparing input for {i}.')
            if self.inputs[i]['generator'] ==  'csv':
                if 'csv_file' in list(self.inputs[i].keys()):

                    self.inputs[i]['csv_file'] = self.inputs[i]['csv_file'].replace("$SNUDDA_DATA", self.data_path)
                    
                    assert os.path.exists(self.inputs[i]['csv_file']), 'csv file does not exist!'
                    
                    if 'random' not in list(self.inputs[i].keys()):
                        if os.path.isdir(self.inputs[i]['csv_file']):
                            print('Using existing csv directory - copying to network directory')
                            shutil.copytree(self.inputs[i]['csv_file'], os.path.join(self.network_path,'input_csvs'), dirs_exist_ok = True)
                            if self.postsynaptic:
                                for n in self.postsynaptic:
                                    self.inputs[i]['csv_file'] = os.path.join(self.inputs[i]['csv_file'], f"{i}_{n}_input.csv")
                        elif os.path.isfile(self.inputs[i]['csv_file']):
                            print('Using existing csv')
                            print(self.inputs[i]['csv_file'])
                            
                        elif 'n_presynaptic' in list(self.inputs[i].keys()):
                            self.n_presynaptic = self.inputs[i]['n_presynaptic']
                            if 'num_inputs' in list(self.inputs[i].keys()):
                                n_inputs_per_connection = self.inputs[i]['num_inputs']
                            #     if isinstance(n_inputs_per_connection, list):
                            #         n_inputs_per_connection = round(self.rng.normal(loc = n_inputs_per_connection[0], 
                            #                                                    scale = n_inputs_per_connection[1]))
                            #     assert isinstance(n_inputs_per_connection, int), 'n_inputs_per_connection must be an integer'
                            # else:
                            #     n_inputs_per_connection = 1

                else: 
                    print('No csv file provided. Generating now.')
                    self.setup_csv(i)
                    
            elif self.inputs[i]['generator'] ==  'poisson':
                print('Using poisson input.')
                
        return
        
    def setup_csv(self, i):
        
        """
        i (str): name of input to setup
        
        """
        input_csv_dir = os.path.join(self.network_path, 'input_csvs')
        if not os.path.exists(input_csv_dir) and self.network_path !=  '/Users/wst/Desktop/Karolinska/Simulation/Neuron/networks/test':
            try:
                os.makedirs(input_csv_dir, exist_ok = True)
            except PermissionError:
                raise PermissionError(f"Cannot create {input_csv_dir}. Check write permissions.")
            except OSError as e:
                raise OSError(f"Failed to create directory: {e}")

        if 'num_inputs' in list(self.inputs[i].keys()):
            n_inputs = self.inputs[i]['num_inputs']
            assert ('num_inputs' in list(self.inputs[i].keys())) and ('frequency' in list(self.inputs[i].keys())), 'number of inputs or frequency not specified!' 
            if isinstance(n_inputs, list):
                rng_num_inputs = np.random.default_rng()
                num_inputs = max(1, int(rng_num_inputs.normal(n_inputs[0], n_inputs[1])))
            else:
                num_inputs = int(n_inputs)
            self.inputs[i]["num_inputs_real"] = num_inputs

        if 'p_soma_synapses' in list(self.inputs[i].keys()):
            p_soma_synapses = self.inputs[i]['p_soma_synapses']
            n_soma_synapses = int(np.rint(p_soma_synapses * num_inputs))
            n_soma_synapses = min(n_soma_synapses, num_inputs)

            self.inputs[i]["num_soma_synapses"] = n_soma_synapses
        
        if 'n_presynaptic' in list(self.inputs[i].keys()):
            self.n_presynaptic = self.inputs[i]['n_presynaptic']
            # if 'num_inputs' in list(self.inputs[i].keys()):
                # n_inputs_per_connection = self.inputs[i]['num_inputs']
                # if isinstance(n_inputs_per_connection, list):
                #     n_inputs_per_connection = round(self.rng.normal(loc = n_inputs_per_connection[0], 
                #                                               scale = n_inputs_per_connection[1]))
                # assert isinstance(n_inputs_per_connection, int), 'n_inputs_per_connection must be an integer'
            # else:
            #     n_inputs_per_connection = 1
                
        else:
            self.n_presynaptic = n_inputs
            # n_inputs_per_connection = n_inputs
            
        if 'start' in list(self.inputs[i].keys()):
            start = self.inputs[i]['start']
        else:
            start = self.default_start
            self.inputs[i]['start'] = start

        if 'end' in list(self.inputs[i].keys()):
            end = self.inputs[i]['end']
        else:
            end = self.default_end
            self.inputs[i]['end'] = end
            
        sigma = self.sigma_map.get(i.split('_')[0], 1)
            
        if 'variability' in self.inputs[i]:
            variability = self.inputs[i]['variability']
            assert isinstance(variability, (float, int)), 'Variability must be numeric.'
        else:
            variability = 1
            
        if 'pauses' in list(self.inputs[i].keys()):
            pauses = self.inputs[i]['pauses']
            assert isinstance(pauses, list), 'Need a start and end time for pause.'
        else:
            pauses = None
            
        if 'bursts' in list(self.inputs[i].keys()):
            bursts = self.inputs[i]['bursts']
            assert isinstance(bursts, list), 'Need a start and end time, and a frequency for burst.'
        else:
            bursts = None
            
        if self.postsynaptic:
            if self.rc is not None:
                print('Running in parallel with ipyparallel.')
                
                # collect tasks for parallel execution
                postsynaptic_configs = []
                for n in self.postsynaptic:
                    # config dict for each task
                    input_config = {
                        'input_type': i,
                        'postsynaptic': n,
                        'frequency': self.inputs[i]['frequency'],
                        'n_presynaptic': self.inputs[i]['n_presynaptic'],
                        'sigma': sigma,
                        'variability': variability,
                        'pauses': pauses,
                        'bursts': bursts,
                        'network_path': self.network_path,
                        'start': start, 
                        'end': end,
                        "num_inputs": num_inputs
                    }
                    postsynaptic_configs.append(input_config)
                self.tasks = postsynaptic_configs
                print(f"Distributing {len(postsynaptic_configs)} neurons across {len(self.d_view)} workers.")
                self.d_view.scatter('postsynaptic_configs', postsynaptic_configs, block=True)
                
                print(f"Processing {len(postsynaptic_configs)} neurons in parallel.")
                
                self.d_view.execute("trains = sg.generate_csv_parallel(postsynaptic_configs)", block=True)
                
                try:                       
                    engines = self.d_view['trains']
                    for engine in engines:
                        for n in engine.keys():
                            self.total_trains[str(n)][i] = engine[n]['train']

                    print(f"Completed parallel execution.")
                except Exception as e:
                    print(f"Error in parallel execution: {e}")
            else:
                # sequential execution
                for n in self.postsynaptic:
                    freq = self.inputs[i]['frequency']
                    n_pre = self.inputs[i]['n_presynaptic']
                    filename, train = self.generate_csv(input_type = i, postsynaptic = n, 
                                    frequency = freq,
                                    n_presynaptic = n_pre, 
                                    sigma = sigma, variability = variability, 
                                    pauses = pauses, bursts = bursts, start = start, end = end, num_inputs = num_inputs)
                    self.total_trains[str(n)][i] = train
        return
    
    def generate_csv(self, input_type = None, postsynaptic = None, frequency = None, n_presynaptic = None, sigma = 0.6, variability = 1, 
                    pauses = None, bursts = None, jitter = 0.02, start= 0, end = 5, num_inputs = None, config = None):
    
        """
        Generate csv files with spike trains
        """
        
        if config is not None: 
            input_type = config['input_type']
            postsynaptic = config['postsynaptic']
            frequency = config['frequency']
            n_presynaptic = config.get('n_presynaptic')
            sigma = config.get('sigma', 0.6)
            variability = config.get('variability', 1)
            pauses = config.get('pauses')
            bursts = config.get('bursts')
            start = config.get('start')
            end = config.get('end')
            num_inputs = config.get('num_inputs')

        rng = np.random.default_rng()
        file_name = os.path.join(self.network_path, 'input_csvs', f"{input_type}_{postsynaptic}_input.csv")
        
        start = start*s
        end = end*s

        f = np.abs(rng.normal(loc = frequency, scale = variability))
        st = list(SLNP(rate = f*Hz, sigma = sigma, t_start = start, t_stop = end).generate_spiketrain().magnitude.flatten())
       
        while len(st) < 1:          ##dirty fix to ensure there is always a spike
            st = list(SLNP(rate = f*Hz, sigma = sigma, t_start = start, t_stop = end).generate_spiketrain().magnitude.flatten())
        
        if bursts:
            for burst in bursts: 
                if len(burst) ==  4:
                    proportion = burst[3]
                    if proportion > 1:
                        proportion/= 100
                    assert 0 <= proportion <= 1, f'Proportion must be within 0 and 1, {proportion} is out of bounds.'
                    
                else:
                    proportion = 1  ## 100% of presynaptic bursting if not otherwise specified
                i = rng.integers(0, n_presynaptic-1)
                if i <  int(np.rint(n_presynaptic * proportion)):
                    
                    burst_jitter_a, burst_jitter_b = rng.uniform(-jitter, jitter, size = 2)
                    
                    f = np.abs(rng.normal(loc = burst[2], scale = variability))
                    burst_train = list(SLNP(rate = f*Hz, sigma = sigma, 
                                           t_start = (burst[0] + burst_jitter_a)*s,                                                   
                                           t_stop = (burst[1] + burst_jitter_b)*s).generate_spiketrain().magnitude.flatten())
                    st = [t for t in st if (t < burst[0] + burst_jitter_a) | 
                                 (t > burst[1] + burst_jitter_b)]
                    st +=  burst_train

            
        # for k in range(n_inputs_per_connection):   
        #     csv_spikes.append(train)
            
        if pauses:
            for pause in pauses: 
                if len(pause) ==  3:
                    proportion = pause[2]
                else:
                    proportion = 1
                i = rng.integers(0, n_presynaptic-1)
                if i in range(int(np.rint(n_presynaptic*proportion))):
                    st = [t for t in st if (t < pause[0]) | (t > pause[1])] 

            
        if self.network_path !=  os.path.join(self.data_path, 'test'):
            try:
                os.makedirs(os.path.join(self.network_path, 'input_csvs'), exist_ok = True)
                with open(file_name, 'w+', newline = '') as csvfile:
                    writer = csv.writer(csvfile, delimiter = ',')
                    csv_spikes = [st.copy() for _ in range(num_inputs)]
                    writer.writerows(csv_spikes)
            except PermissionError:
                raise PermissionError(f"Cannot write to {file_name}. Check write permissions.")
            except IOError as e:
                raise IOError(f"Failed to write csv file {file_name}: {e}")
            except Exception as e:
                raise Exception(f"Unexpected error writing csv for {input_type}: {e}")
                
        return file_name, st
    

    def generate_csv_parallel(self, configs):
        
        assert isinstance(configs, list), "Ensure a list of arguments are passed to generate_csv_parallel"
        parallel_trains = {}
        for config in configs:
            file_name, train = self.generate_csv(config = config)
            parallel_trains[config['postsynaptic']] = {'file_name':file_name, 'train': train}
        return parallel_trains
        
        
    def import_csv_train(self, csv_file):
        """import spike train data from a csv file"""
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"csv file not found: {csv_file}")
            
        return [np.sort(np.fromstring(row, sep = ',', dtype = float)) for row in open(csv_file, "r")]
    
    def random_csv(self, postsynaptic, input_type, signal):
        """select a random csv file from a given directory"""
        try:
            if signal and 'signal_csv_file' in self.inputs[input_type]:
                if 'p_signal' in self.inputs[input_type]:
                    if self.rng.random() < float(self.inputs[input_type]['p_signal']):
                        csv_dir = self.inputs[input_type]['signal_csv_file']
                    else:
                        csv_dir = self.inputs[input_type]['csv_file']
                else:
                    csv_dir = self.inputs[input_type]['signal_csv_file']
            else:
                csv_dir = self.inputs[input_type]['csv_file']
            
            #cache files for repeated access
            if os.path.isdir(csv_dir):
                if not hasattr(self, '_csv_cache'):
                    self._csv_cache = {}
                    
                if csv_dir not in self._csv_cache:
                    self._csv_cache[csv_dir] = [
                        os.path.join(csv_dir, entry.name) 
                        for entry in os.scandir(csv_dir) 
                        if entry.is_file() and entry.name.endswith('.csv')
                    ]
                
                # get cached list of csv files
                csv_files = self._csv_cache[csv_dir]
                file_name = self.rng.choice(csv_files)
                
                return file_name
            else:
                return self.inputs[input_type]['signal_csv_file']
        except (KeyError, FileNotFoundError):
            pass
        
        return ''
    
    def write_json(self):
        if not self.postsynaptic:
            # simple case: write inputs directly if no postsynaptic neurons
            with open(self.file_name, 'w') as f:
                json.dump({self.neuron_class: self.inputs}, f, indent = 4)
            return
    
        self.output = {}
        
        if self.rc is not None:
            d_view = self.d_view
            
            print('Running in parallel')
            
            d_view.scatter("neuron_idx", self.postsynaptic, block = True)
            d_view.push({"inputs":self.inputs}, block = True)

            cmd_str = ("sg.process_neuron_parallel(neurons = neuron_idx, inputs = inputs)")
            d_view.execute(cmd_str, block = True)
            output_list = d_view.gather("sg.output", block = True)
            self.output = {k: v for d in output_list for k, v in d.items()} 
        else:
            for n in self.postsynaptic:
                if n in self.signals:
                    self.output[str(n)] = {
                        input_source: self._process_input_source(n, input_source, signal = True) 
                        for input_source in self.inputs
                    }
                else:
                    self.output[str(n)] = {
                        input_source: self._process_input_source(n, input_source, signal = False) 
                        for input_source in self.inputs
                    }
            
        self._write_output_file()
        
        return
    
    def process_neuron_parallel(self, neurons, inputs):
        
        self.output = {}
        signals = []
        for n in neurons:
            if n in signals:
                signal = True
            else:
                signal = False
            self.output[str(n)] = {input_source: self._process_input_source(n, input_source, signal = signal) for input_source in inputs}

        return 

    def _process_input_source(self, postsynaptic, input_source, signal = False):
    
        input_config = copy.deepcopy(self.inputs[input_source])

        if 'random' in input_config:
            csv_file = self.random_csv(postsynaptic = postsynaptic, input_type = input_source, signal = signal)
        elif 'csv_file' in input_config:
            csv_file = input_config['csv_file']
        else:
            csv_file = os.path.join(
                self.network_path, 
                'input_csvs', 
                f'{input_source}_{postsynaptic}_input.csv'
            )
        input_config['csv_file'] = csv_file

        if 'dendrite_location' in input_config:
            d_locs = input_config['dendrite_location'][str(postsynaptic)]
            input_config['dendrite_location'] = d_locs

        return input_config
    
    def _write_output_file(self):
        
        if os.path.exists(self.file_name):
            if self.merge_okay == True:
                print('Merging with existing inputs')
                
                with open(self.file_name, 'r') as f:
                    existing_inputs = json.load(f)
                
                combined_inputs = {**existing_inputs, **self.output}
                
                with open(self.file_name, 'w') as f:
                    json.dump(combined_inputs, f, indent = 4, cls = NpEncoder)
            else:
                print('Output file already exists, and automatic merging is currently off. Double check before proceeding.')
        else:
            with open(self.file_name, 'w') as f:
                json.dump(self.output, f, indent = 4, cls = NpEncoder)
        return
    
    
    def get_unique_trains(self, postsynaptic):
        """
        returns unique spiketrains
        
        """
        unique_trains = []
        for train in self.total_spikes[str(postsynaptic)].values():
            unique_trains.append(train[0])
        return unique_trains
    
    
    def add_burst(self, postsynaptic, input_type, start, end, frequency, proportion = 1, sigma = 0.6, jitter = 0.02, variability = 1): 
        """add burst to existing spiketrain"""
        
        if not os.path.exists(os.path.join(self.network_path,'input_csvs')):
            os.mkdir(os.path.join(self.network_path,'input_csvs'))
            
        if not isinstance(postsynaptic, list):
            postsynaptic = [postsynaptic]
            
        for n in postsynaptic: 
            cell_spikes = self.total_spikes[str(n)]
            for k in cell_spikes.keys():
                if input_type in k:
                    if np.random.default_rng().uniform() < proportion:
                        
                        burst_jitter_a, burst_jitter_b = self.rng.uniform(-jitter, jitter, size = 2)

                        pre_spikes = cell_spikes[k]
                        start = start + burst_jitter_a
                        end = end + burst_jitter_b
                        f = np.abs(np.random.default_rng().normal(loc = frequency, scale  = variability))
                        burst_train = list(SLNP(rate = f*Hz, sigma = sigma, t_start = (start)*s,t_stop = (end)*s).generate_spiketrain().magnitude.flatten())
                        kept_spikes = [spike for spike in pre_spikes[0] if (spike < start) | (spike > end)]
                        kept_spikes +=  burst_train
                        self.total_spikes[str(n)][k] = [sorted(kept_spikes)]*len(pre_spikes)
                        
                        try: 
                            csv_file_name = os.path.join(self.network_path,'input_csvs', str(k) +'_' + str(n) +'_input.csv')

                            with open(csv_file_name, 'w+', newline = '') as csvfile:
                                writer = csv.writer(csvfile, delimiter = ',')
                                writer.writerows(self.total_spikes[str(n)][k])
            
                            self.output[str(n)][k]['csv_file'] = csv_file_name
                        except (IOError, PermissionError) as e:
                            raise IOError(f"Failed to write csv for {k}: {e}")
    
        with open(self.file_name, 'w+') as f:
            json.dump(self.output, f, indent = 4, cls = NpEncoder)
        return
    
    def use_default_inputs(self):
        """this file could be configured for different uses, currently SNr specific"""
        
        file_path = Path(__file__).parent.parent / 'data' / 'default_inputs.json'
        with open(file_path) as f:
            self.defaults = json.load(f)
        return

    
    def calculate_fr_cv(self, postsynaptic = 0):
        """"Calculate firing rate and CV_ISI of spike trains"""
        frs = {}
        cvs = {}
    
        for input_name, spiketrain in self.total_trains[str(postsynaptic)].items():
            start = self.inputs[input_name]['start']
            end = self.inputs[input_name]['end']
            spiketrain = [x for x in spiketrain if (x < end) & (x> start)]
            st = SpikeTrain(times = spiketrain*s, t_start = start*s, t_stop =end*s)
            
            frs[input_name] = len(spiketrain)/(end -start)
            cvs[input_name]  = ele_stats.cv(ele_stats.isi(st))

        return {'firing_rate': frs, 'CV_isi': cvs}

    
#%%

if __name__ ==  "__main__":
    
    net_name = 'test'
    network_path = os.path.join(os.getcwd(), "networks", net_name)
    try:
        os.mkdir(network_path)
    except FileExistsError:
        pass 
    

    ray_parallel = False
    if ray_parallel:
        import ray
        ray.shutdown()
        ray.init(num_cpus = 7)
    
        if ray.is_initialized():
            ray_parallel = True
        else:
            ray_parallel = False
            print('Ray not initialised, running in serial.')
    
    dummy_network = list(np.arange(0, 10))
    sg = Signaux(network_path, ray_parallel = ray_parallel,  postsynaptic = dummy_network, input_defs = [{'dSPN': {'generator': 'csv','n_presynaptic':10,"num_inputs" : 50, 'frequency': 1.4, 'variability' : 1,'end':5, 'bursts': [[1.25,1.5, 50, 100]]}}])
    sg.csv_inputs()
    sg.write_json()
        
    if ray_parallel:
        ray.shutdown()


