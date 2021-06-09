import matplotlib.pyplot as plt
import numpy as np


def plot_data(data, title, start=0, end=1250, bool_time_major=True):
    if bool_time_major:
        data = data.transpose(1, 0)
    else:
        pass
    for i in range(len(data)):
        plt.plot(data[i][start:end])
    plt.title(title)


def np_relu(arr):
    _arr = np.array([], dtype=arr[0].dtype)
    for i in arr:
        if i > 0:
            _arr = np.append(_arr, i)
        else:
            _arr = np.append(_arr, 0)
    return _arr


def compatibility_density(cpt, w_s):
    half_w_s = int(w_s / 2)
    _len = len(cpt)
    _score = np_relu(cpt)
    _score = np.array([], dtype=cpt[0].dtype)
    _density = np.array([], dtype=float)
    for i in range(_len):
        if i <= half_w_s:
            current_w_s = half_w_s + i
            _score = np.append(_score, np.sum(cpt[0:current_w_s]))
            _density = np.append(_density, np.sum(cpt[0:current_w_s]) / current_w_s)
        elif half_w_s < i < (_len - half_w_s):
            current_w_s = w_s
            _score = np.append(_score, np.sum(cpt[i - half_w_s:i + half_w_s]))
            _density = np.append(_density, np.sum(cpt[i - half_w_s:i + half_w_s]) / current_w_s)
        else:
            current_w_s = half_w_s + _len - i
            _score = np.append(_score, np.sum(cpt[i - half_w_s:_len]))
            _density = np.append(_density, np.sum(cpt[i - half_w_s:_len]) / current_w_s)
    _range = np.max(_density) - np.min(_density)
    return (_density - np.min(_density)) / _range


def plot_att(data, title, cpt1, cpt2, cpt3, plot_type='line', bool_time_major=True):
    if bool_time_major:
        data = data.transpose(1, 0)
    else:
        pass

    if plot_type == 'bar':
        plt.subplot(411)
        plt.title(title)
        for i in range(len(data)):
            plt.plot(data[i])
        plt.subplot(412)
        plt.title('cpt1')
        plt.bar(np.arange(len(cpt1)), cpt1)
        plt.subplot(413)
        plt.title('cpt2')
        plt.bar(np.arange(len(cpt2)), cpt2)
        plt.subplot(414)
        plt.title('cpt3')
        plt.bar(np.arange(len(cpt3)), cpt3)
        plt.show()
    else:
        plt.subplot(411)
        plt.title(title)
        for i in range(len(data)):
            plt.plot(data[i])
        plt.subplot(412)
        plt.title('cpt1')
        plt.plot(cpt1)
        plt.subplot(413)
        plt.title('cpt2')
        plt.plot(cpt2)
        plt.subplot(414)
        plt.title('cpt3')
        plt.plot(cpt3)
        plt.show()

