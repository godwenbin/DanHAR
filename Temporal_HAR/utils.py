import numpy as np
import pandas as pd

def load_data(which='SG'):
    if which == 'PAMAP2_new':
        train_x = np.load('PAMAP2_new/train_X_new.npy')[:, :,0:9]
        y_train = np.load('PAMAP2_new/train_y_new.npy')
        train_y = np.asarray(pd.get_dummies(y_train), dtype=np.int8)  # one-hot val labels
        test_x = np.load('PAMAP2_new/total_pamap2_valtestx.npy')[:, :,0:9]
        y_test = np.load('PAMAP2_new/total_pamap2_valtesty.npy')
        test_y = np.asarray(pd.get_dummies(y_test), dtype=np.int8)  # one-hot val labels
        print('# PAMAP2_new  data loaded.')
        return train_x, train_y, test_x, test_y
    elif which == 'UCI_3axis':
        train_x = np.load('UCI-HAR-Dataset/processed/np_train_x.npy')[:, :,0:3]
        train_y = np.load('UCI-HAR-Dataset/processed/np_train_y.npy')
        test_x = np.load('UCI-HAR-Dataset/processed/np_test_x.npy')[:, :, 0:3]
        test_y = np.load('UCI-HAR-Dataset/processed/np_test_y.npy')
        print('#  UCI HAR 3axis data loaded.')
        return train_x, train_y, test_x, test_y
    else:
        pass


def shuffle(data, labels):
    index = np.arange(len(data))
    np.random.shuffle(index)
    return data[index], labels[index]


def cross_val(data, labels, epoch, cross=5):
    l = len(data)
    i = epoch % cross
    j = np.ceil(l / cross)
    start_idx = int(i * j)
    end_idx = int((i + 1) * j)
    train_xc = np.append(data[:start_idx], data[end_idx:], axis=0)
    train_yc = np.append(labels[:start_idx], labels[end_idx:], axis=0)
    val_xc = data[start_idx:end_idx]
    val_yc = labels[start_idx:end_idx]
    return train_xc, train_yc, val_xc, val_yc


def extract_batch_size(_train, epoch, batch_size):
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)
    for i in range(batch_size):
        index = ((epoch - 1) * batch_size + i) % len(_train)
        batch_s[i] = _train[index]
    return batch_s




