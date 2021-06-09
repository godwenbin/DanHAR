import numpy as np
from collections import Counter
import torch
def cut(dataset_name, ratio=(7, 1, 2), ifsave=False):
    X = np.vstack((np.load(dataset_name+'/x_train.npy'), np.load(dataset_name+'/x_test.npy'))).tolist()
    Y = np.hstack((np.load(dataset_name+'/y_train.npy'), np.load(dataset_name+'/y_test.npy'))).tolist()
    X_eval, X_test, Y_eval, Y_test = [], [], [], []
    test_ratio, eval_ratio = ratio[2]/sum(ratio), ratio[1]/sum(ratio)
    y_dict = Counter(Y)
    for each_category in y_dict.keys():
        init_num = y_dict[each_category]
        eval_num = int(init_num * eval_ratio)
        test_num = int(init_num * test_ratio)
        while eval_num + test_num:
            x_elem = X[Y.index(each_category)]
            X.remove(x_elem)
            Y.remove(each_category)
            if eval_num > 0:
                X_eval.append(x_elem)
                Y_eval.append(each_category)
                eval_num -= 1
            else:
                X_test.append(x_elem)
                Y_test.append(each_category)
                test_num -= 1

    X_train = np.array(X, dtype=np.float32)
    X_eval = np.array(X_eval, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    Y_train = np.array(Y, dtype=np.int64)
    Y_eval = np.array(Y_eval, dtype=np.int64)
    Y_test = np.array(Y_test, dtype=np.int64)

    print(X_train.shape, X_eval.shape, X_test.shape, Y_train.shape, Y_eval.shape, Y_test.shape)
    print('（训练，验证，测试）三个标签集的类别种类数分别为【%d， %d， %d】' % (len(Counter(Y_train)), len(Counter(Y_eval)), len(Counter(Y_test))))
    if ifsave:
        np.save(dataset_name+'/x_train_new', X_train)
        np.save(dataset_name+'/x_eval_new', X_eval)
        np.save(dataset_name+'/x_test_new', X_test)
        np.save(dataset_name+'/y_train_new', Y_train)
        np.save(dataset_name+'/y_eval_new', Y_eval)
        np.save(dataset_name+'/y_test_new', Y_test)

dataset_list = ['OPPORTUNITY', 'PAMAP2', 'UCI', 'UNIMIB', 'WISDM', 'USC_HAD']
for each_dataset in dataset_list:
    cut(each_dataset, ratio=(7, 1, 2), ifsave=True)