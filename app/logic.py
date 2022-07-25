import json
import os
import pickle
import threading
import time
import yaml

import pandas as pd
import numpy as np

from distutils import dir_util


class AppLogic:

    def __init__(self):
        # === Status of this app instance ===

        # Indicates whether there is data to share, if True make sure self.data_out is available
        self.status_available = False

        # Only relevant for master, will stop execution when True
        self.status_finished = False

        # === Parameters set during setup ===
        self.id = None
        self.master = None
        self.clients = None

        # === Data ===
        self.data_incoming = []
        self.data_outgoing = None

        # === Internals ===
        self.thread = None
        self.iteration = 0
        self.progress = 'not started yet'

        # === Custom ===
        self.train = None
        self.sep = None
        self.label_column = None
        self.data = None
        self.data_without_label = None
        self.mode = None

        self.filename = None
        self.values = None

        self.threshold = None
        self.number_of_features = None

        self.to_drop = None

    def handle_setup(self, client_id, master, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.master = master
        self.clients = clients
        print(f'Received setup: {self.id} {self.master} {self.clients}')

        self.read_config()

        self.thread = threading.Thread(target=self.app_flow)
        self.thread.start()

    def handle_incoming(self, data):
        print(f'Received data: {data}')
        # This method is called when new data arrives
        self.data_incoming.append(data.read())

    def handle_outgoing(self):
        print(f'Submit data: {self.data_outgoing}')
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def read_config(self):
        dir_util.copy_tree('/mnt/input/', '/mnt/output/')
        with open('/mnt/input/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_feature_reduction']
            self.train = config['files']['input']
            self.sep = config['files']['sep']
            self.label_column = config['files']['label_column']
            self.mode = config.get('mode', 'pearson')
            self.threshold = config.get('threshold', 0.95)
            self.number_of_features = config.get('features')

    def app_flow(self):
        # This method contains a state machine for the slave and master instance

        # === States ===
        state_initializing = 1
        state_read_input = 2
        state_compute_local = 2
        state_gather = 3
        state_wait = 4
        state_result_ready = 5
        state_finishing = 6

        # Initial state
        state = state_initializing
        self.progress = 'initializing...'

        while True:

            if state == state_initializing:
                if self.id is not None:  # Test if setup has happened already
                    state = state_read_input

            # COMMON PART

            if state == state_read_input:
                print('Reading input...')
                self.data = pd.read_csv(os.path.join(f'/mnt/input/', self.train), sep=self.sep)
                self.data_without_label = self.data.drop(self.label_column, axis=1)
                print('Read input.')
                state = state_compute_local

            if state == state_compute_local:
                print('Calculate local values...')
                self.values = self.data_without_label.to_numpy()
                local_data_1 = None  # Contains sample size
                local_data_2 = None  # Contains sum and sum of squares
                local_data_3 = None  # Contains sum of pairwise products
                if self.mode == 'pearson':
                    local_data_1 = self.values.shape[0]
                    local_data_2 = np.zeros((self.values.shape[1], 2))
                    local_data_2[:, 0] = np.sum(np.square(self.values), axis=0)
                    local_data_2[:, 1] = np.sum(self.values, axis=0)
                    local_data_3 = self.values.T @ self.values
                print(f'Calculated local values: {local_data_1} {local_data_2} {local_data_3}')

                data = pickle.dumps({
                    'data_1': local_data_1,
                    'data_2': local_data_2,
                    'data_3': local_data_3,
                })

                if self.master:
                    self.data_incoming.append(data)
                    state = state_gather
                else:
                    self.data_outgoing = data
                    self.status_available = True
                    state = state_wait

            if state == state_result_ready:
                result_df = self.data_without_label.drop(columns=self.to_drop)
                result_df[self.label_column] = self.data.loc[:, self.label_column]
                result_df.to_csv(os.path.join('/mnt/output/', self.train), index=False, sep=self.sep)

                if self.master:
                    state = state_finishing
                else:
                    break

            # GLOBAL PART

            if state == state_gather:
                if len(self.data_incoming) == len(self.clients):
                    global_data_1 = np.zeros((1,))
                    global_data_2 = np.zeros((self.values.shape[1], 2))
                    global_data_3 = np.zeros((self.values.shape[1], self.values.shape[1]))

                    for local_bytes in self.data_incoming:
                        data = pickle.loads(local_bytes)
                        local_data_1 = data['data_1']  # Contains sample size
                        local_data_2 = data['data_2']  # Contains sum and sum of squares
                        local_data_3 = data['data_3']  # Contains sum of pairwise products

                        if self.mode == 'pearson':
                            global_data_1 += local_data_1
                            global_data_2 += local_data_2
                            global_data_3 += local_data_3

                    if self.mode == 'pearson':
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

                        print(f'Correlations: {correlations}')

                        # Create correlation matrix
                        corr_matrix = pd.DataFrame(data=np.abs(correlations),
                                                   index=None,
                                                   columns=self.data_without_label.columns)

                        # Select upper triangle of correlation matrix
                        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

                        if self.number_of_features is not None:
                            max_corr = [(column, max(upper[column])) for column in upper.columns]
                            max_corr.sort(key=lambda x: x[1])
                            to_drop = max_corr[self.number_of_features:]
                            self.to_drop = [td[0] for td in to_drop]
                        elif self.threshold is not None:
                            self.to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]

                        # Drop features
                        print(f'To drop: {self.to_drop}')

                        self.data_outgoing = pickle.dumps({
                            'to_drop': self.to_drop,
                        })

                    self.status_available = True
                    state = state_result_ready
                else:
                    print(f'Have {len(self.data_incoming)} of {len(self.clients)} so far, waiting...')

            if state == state_finishing:
                time.sleep(10)
                self.status_finished = True
                break

            # LOCAL PART

            if state == state_wait:
                if len(self.data_incoming) > 0:
                    pkg = pickle.loads(self.data_incoming[0])
                    if self.mode == 'pearson':
                        self.to_drop = pkg['to_drop']
                        print(f'To drop: {self.to_drop}')

                    state = state_result_ready

            time.sleep(1)


logic = AppLogic()
