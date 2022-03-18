import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

COLUMN_NAMES = [
    'user',
    'activity',
    'timestamp',
    'x-axis',
    'y-axis',
    'z-axis'
]

LABELS = [
    'Downstairs',
    'Jogging',
    'Sitting',
    'Standing',
    'Upstairs',
    'Walking'
]

DATA_PATH = 'WISDM_ar_v1.1_raw.txt'

RANDOM_SEED = 13

# Data preprocessing
TIME_STEP = 100

# Model
N_CLASSES = 6
N_FEATURES = 3  # x-acceleration, y-acceleration, z-acceleration

# Hyperparameters optimized
SEGMENT_TIME_SIZE = 200


if __name__ == '__main__':

    # LOAD DATA
    data = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)
    data['z-axis'].replace({';': ''}, regex=True, inplace=True)
    data = data.dropna()

    # SHOW GRAPH FOR JOGGING
    data[data['activity'] == 'Jogging'][['x-axis']][:50].plot(subplots=True, figsize=(16, 12), title='Jogging')
    plt.xlabel('Timestep')
    plt.ylabel('X acceleration (dg)')

    # SHOW ACTIVITY GRAPH
    activity_type = data['activity'].value_counts().plot(kind='bar', title='Activity type')
    # plt.show()

    # DATA PREPROCESSING
    data_convoluted = []
    labels = []

    # Slide a "SEGMENT_TIME_SIZE" wide window with a step size of "TIME_STEP"
    for i in range(0, len(data) - SEGMENT_TIME_SIZE, TIME_STEP):
        x = data['x-axis'].values[i: i + SEGMENT_TIME_SIZE]
        y = data['y-axis'].values[i: i + SEGMENT_TIME_SIZE]
        z = data['z-axis'].values[i: i + SEGMENT_TIME_SIZE]
        data_convoluted.append([x, y, z])

        # Label for a data window is the label that appears most commonly
        label = stats.mode(data['activity'][i: i + SEGMENT_TIME_SIZE])[0][0]
        labels.append(label)

    # Convert to numpy
    data_convoluted = np.asarray(data_convoluted, dtype=np.float32).transpose(0, 2, 1)

    # One-hot encoding
    labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
    print('-'*50)
    print("Convoluted data shape: ", data_convoluted.shape)
    print("Labels shape:", labels.shape)

    # SPLIT INTO TRAINING AND TEST SETS
    x_train, x_test, y_train, y_test = train_test_split(data_convoluted, labels, test_size=0.3, random_state=10)
    np.save('train_x21.npy', x_train)
    np.save('train_y21.npy', y_train)
    np.save('test_x21.npy', x_test)
    np.save('test_y21.npy', y_test)

    print('-'*50)
    print("x train size: ", x_train.shape)
    print("x test size: ", x_test.shape)
    print("y train size: ", y_train.shape)
    print("y test size: ", y_test.shape)

