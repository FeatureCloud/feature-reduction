import os

import numpy as np
import pandas as pd

if __name__ == '__main__':
    label_column = 'c'
    data = pd.read_csv(os.path.join('../sample_data/data1/input.csv'), sep=',')
    data_without_label = data.drop(label_column, axis=1)
    values = data_without_label.to_numpy()

    local_data_1 = values.shape[0]
    local_data_2 = np.zeros((values.shape[1], 2))
    local_data_2[:, 0] = np.sum(np.square(values), axis=0)
    local_data_2[:, 1] = np.sum(values, axis=0)
    local_data_3 = values.T @ values

    global_data_1 = local_data_1
    global_data_2 = local_data_2
    global_data_3 = local_data_3

    global_mean_square = global_data_2[:, 0] / global_data_1
    global_mean = global_data_2[:, 1] / global_data_1
    global_mean_pair = global_data_3 / global_data_1
    global_variance = global_mean_square - np.square(global_mean)
    global_stddev = np.sqrt(global_variance)

    print(np.var(values, axis=0))
    print(global_variance)
    print(np.cov(values.T, ddof=0))

    global_pair = global_mean.reshape(global_mean.shape[0], 1) @ global_mean.reshape(1, global_mean.shape[0])
    global_stddev_pair = global_stddev.reshape(global_stddev.shape[0], 1) @ \
                         global_stddev.reshape(1, global_stddev.shape[0])
    covariances = global_mean_pair - global_pair
    correlations = covariances / global_stddev_pair

    corr_matrix = pd.DataFrame(data=np.abs(correlations), index=None, columns=data_without_label.columns)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    print(data_without_label.corr())
    print(corr_matrix)

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    print(to_drop)
