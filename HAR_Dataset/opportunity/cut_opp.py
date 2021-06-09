import warnings
warnings.filterwarnings('ignore')

import numpy as np
import _pickle as cp
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics

import torch
from torch import nn
import torch.nn.functional as F

from thop import profile
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

NB_SENSOR_CHANNELS = 113
#64 8;60 30;30 15
SLIDING_WINDOW_LENGTH = 30
SLIDING_WINDOW_STEP = 15
NUM_CLASSES = 18

import os
import sys

# add the 'src' directory as one where we can import modules
src_dir = os.path.join(os.getcwd(), os.pardir, 'src')
sys.path.append(src_dir)

from sliding_window import sliding_window


def load_dataset(filename):
    with open(filename, 'rb') as f:
        data = cp.load(f)

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test


print("Loading data...")
X_train, y_train, X_test, y_test = load_dataset('./oppChallenge_gestures.data')


assert NB_SENSOR_CHANNELS == X_train.shape[1]
def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

# Sensor data is segmented using a sliding window mechanism
X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)

# Data is reshaped
X_train = X_train.reshape((-1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS)) # for input to Conv1D
X_test = X_test.reshape((-1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS)) # for input to Conv1D
y_train = convert_to_one_hot(y_train,NUM_CLASSES) # one-hot encoding
y_test = convert_to_one_hot(y_test,NUM_CLASSES) # one-hot encoding
print(" ..after sliding and reshaping, train data: inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
print(" ..after sliding and reshaping, test data : inputs {0}, targets {1}".format(X_test.shape, y_test.shape))
np.save("/mnt/ba3b04da-ce1b-4c21-ad1b-3aff7d337cdf/gaowenbin/Project/DeepConvLSTM/opp_dataset_1s/opp_train_x_1s.npy",X_train)
np.save("/mnt/ba3b04da-ce1b-4c21-ad1b-3aff7d337cdf/gaowenbin/Project/DeepConvLSTM/opp_dataset_1s/opp_train_y_1s.npy",y_train)
np.save("/mnt/ba3b04da-ce1b-4c21-ad1b-3aff7d337cdf/gaowenbin/Project/DeepConvLSTM/opp_dataset_1s/opp_test_x_1s.npy",X_test)
np.save("/mnt/ba3b04da-ce1b-4c21-ad1b-3aff7d337cdf/gaowenbin/Project/DeepConvLSTM/opp_dataset_1s/opp_test_y_1s.npy",y_test)
