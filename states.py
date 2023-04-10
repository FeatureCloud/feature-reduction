import json
import os
import pickle
import threading
import time
import yaml

import pandas as pd
import numpy as np

from distutils import dir_util

from FeatureCloud.app.engine.app import AppState, app_state, Role

@app_state('initial', Role.BOTH)
class InitialState(AppState):
    """
    Initialize client.
    """

    def register(self):
        self.register_transition('read input', Role.BOTH)
        
    def run(self) -> str or None:
        self.log("[CLIENT] Initializing")
        if self.id is not None:  # Test if setup has happened already
            self.log(f"[CLIENT] Coordinator {self.is_coordinator}")
        
        return 'read input'

# COMMON PART
@app_state('read input', Role.BOTH)
class ReadInputState(AppState):
    """
    Read input data and config file.
    """

    def register(self):
        self.register_transition('compute local', Role.BOTH)
        self.register_transition('read input', Role.BOTH)
        
    def run(self) -> str or None:
        self.log("Reading input...")
        self.read_config()

        data = pd.read_csv(os.path.join(f'/mnt/input/', self.load('train')), sep=self.load('sep'))
        self.store('data', data)
        data_without_label = data.drop(self.load('label_column'), axis=1)
        self.store('data_without_label', data_without_label)
        self.log("Read input.")
        return 'compute local'

    def read_config(self):
        dir_util.copy_tree('/mnt/input/', '/mnt/output/')
        with open('/mnt/input/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_feature_reduction']
            self.store('train', config['files']['input'])
            self.store('sep', config['files']['sep'])
            self.store('label_column', config['files']['label_column'])
            self.store('mode', config.get('mode', 'pearson'))
            self.store('threshold', config.get('threshold', 0.95))
            self.store('number_of_features', config.get('features'))


@app_state('compute local', Role.BOTH)
class ComputeLocalState(AppState):
    """
    Perform local computation and send the computation data to the coordinator.
    """

    def register(self):
        self.register_transition('gather', Role.COORDINATOR)
        self.register_transition('wait', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        self.log('Calculate local values...')
        values = self.load('data_without_label').to_numpy()
        self.store('values', values)
        local_data_1 = None  # Contains sample size
        local_data_2 = None  # Contains sum and sum of squares
        local_data_3 = None  # Contains sum of pairwise products
        if self.load('mode') == 'pearson':
            local_data_1 = values.shape[0]
            local_data_2 = np.zeros((values.shape[1], 2))
            local_data_2[:, 0] = np.sum(np.square(values), axis=0)
            local_data_2[:, 1] = np.sum(values, axis=0)
            local_data_3 = values.T @ values
        self.log(f'Calculated local values: {local_data_1} {local_data_2} {local_data_3}')

        data = pickle.dumps({
            'data_1': local_data_1,
            'data_2': local_data_2,
            'data_3': local_data_3,
        })

        self.send_data_to_coordinator(data)

        if self.is_coordinator:
            return 'gather'
        else:
            return 'wait'

@app_state('result ready', Role.BOTH)
class ResultReadyState(AppState):
    """
    Writes the results of the global computation.
    """

    def register(self):
        self.register_transition('terminal', Role.BOTH)
        
    def run(self) -> str or None:

        result_df = self.load('data_without_label').drop(columns=self.load('to_drop'))
        result_df[self.load('label_column')] = self.load('data').loc[:, self.load('label_column')]
        result_df.to_csv(os.path.join('/mnt/output/', self.load('train')), index=False, sep=self.load('sep'))

        return 'terminal'

# GLOBAL PART
@app_state('gather', Role.COORDINATOR)
class GatherState(AppState):
    """
    The coordinator receives the local computation data from each client and aggregates it.
    The coordinator broadcasts the global computation data to the clients.
    """

    def register(self):
        self.register_transition('result ready', Role.COORDINATOR)
        
    def run(self) -> str or None:
        data_incoming = self.gather_data()

        global_data_1 = np.zeros((1,))
        global_data_2 = np.zeros((self.load('values').shape[1], 2))
        global_data_3 = np.zeros((self.load('values').shape[1], self.load('values').shape[1]))

        for local_bytes in data_incoming:
            data = pickle.loads(local_bytes)
            local_data_1 = data['data_1']  # Contains sample size
            local_data_2 = data['data_2']  # Contains sum and sum of squares
            local_data_3 = data['data_3']  # Contains sum of pairwise products

            if self.load('mode') == 'pearson':
                global_data_1 += local_data_1
                global_data_2 += local_data_2
                global_data_3 += local_data_3

        if self.load('mode') == 'pearson':
            global_mean_square = global_data_2[:, 0] / global_data_1
            global_mean = global_data_2[:, 1] / global_data_1
            global_mean_pair = global_data_3 / global_data_1
            global_variance = global_mean_square - np.square(global_mean)
            global_stddev = np.sqrt(global_variance)

            global_pair = global_mean.reshape(global_mean.shape[0], 1) @ \
                                      global_mean.reshape(1, global_mean.shape[0])
            global_stddev_pair = global_stddev.reshape(global_stddev.shape[0], 1) @ \
                                global_stddev.reshape(1, global_stddev.shape[0])
            covariances = global_mean_pair - global_pair
            correlations = covariances / global_stddev_pair

            self.log(f'Correlations: {correlations}')

            # Create correlation matrix
            corr_matrix = pd.DataFrame(data=np.abs(correlations),
                                                   index=None,
                                                   columns=self.load('data_without_label').columns)

            # Select upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            if self.load('number_of_features') is not None:
                max_corr = [(column, max(upper[column])) for column in upper.columns]
                max_corr.sort(key=lambda x: x[1])
                to_drop = max_corr[self.load('number_of_features'):]
                to_drop = [td[0] for td in to_drop]
                self.store('to_drop', to_drop)
            elif self.load('threshold') is not None:
                to_drop = [column for column in upper.columns if any(upper[column] > self.load('threshold'))]
                self.store('to_drop', to_drop)
            # Drop features
            self.log(f"To drop: {self.load('to_drop')}")
            self.broadcast_data(pickle.dumps({
                'to_drop': self.load('to_drop'),
            }), send_to_self=False)

        return 'result ready'


# LOCAL PART
@app_state('wait', Role.PARTICIPANT)
class WaitState(AppState):
    """
    The participant waits until it receives the aggregation data from the coordinator.
    """

    def register(self):
        self.register_transition('result ready', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        data = self.await_data()
        pkg = pickle.loads(data)

        if self.load('mode') == 'pearson':
            self.store('to_drop', pkg['to_drop'])
            self.log(f"To drop: {self.load('to_drop')}")

        return 'result ready'
