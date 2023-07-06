import numpy as np
import os
from optim.Optimizer import *

def RMSE(h, y):
    if len(h.shape) > 1:
        h = h.squeeze()
    se = np.square(h - y)
    mse = np.mean(se)
    rmse = np.sqrt(mse)
    return rmse

def optimizer(optim_name, gamma):
    if optim_name == 'SGD':
        optim = SGD()
    elif optim_name == 'Momentum':
        optim = Momentum(gamma)
    else:
        raise NotImplementedError
    return optim


def load_data(data_name, normalize):
    path = os.path.join('./data', data_name)

    if data_name == 'CCPP':
        train_x, train_y = CCPPData(path, 'train.csv', normalize)
        test_x, test_y = CCPPData(path, 'test.csv', normalize)

    elif data_name == 'Airbnb':
        train_x, train_y = AirbnbData(path, 'train.csv', normalize)
        test_x, test_y = AirbnbData(path, 'test.csv', normalize)

    elif data_name == 'RealEstate':
        train_x, train_y = RealEstateData(path, 'train.csv', normalize)
        test_x, test_y = RealEstateData(path, 'test.csv', normalize)

    else:
        raise NotImplementedError

    return (train_x, train_y), (test_x, test_y)


def AirbnbData(path, filename, normalize):
    return load_reg_data(path, filename, target_at_front=False, normalize=normalize)

def RealEstateData(path, filename, normalize):
    return load_reg_data(path, filename, target_at_front=False, normalize=normalize)

def CCPPData(path, filename, normalize):
    return load_reg_data(path, filename, target_at_front=False, normalize=normalize)


def load_reg_data(path, filename, target_at_front, normalize=None, shuffle=False):
    fullpath = os.path.join(path, filename)

    with open(fullpath, 'r') as f:
        lines = f.readlines()
    lines = [s.strip().split(',') for s in lines]

    data = lines[1:]

    data = np.array([[float(f) for f in d] for d in data], dtype=np.float64)
    if target_at_front:
        x, y = data[:, 1:], data[:, 0]
    else:
        x, y = data[:, :-1], data[:, -1]

    num_data = x.shape[0]
    if normalize == 'MinMax':
        # ====== EDIT HERE ======
        x = x.astype(np.float64)
        for i in range(x.shape[1]):
            new = (x[:, i] - np.min(x[:, i])) / (np.max(x[:, i]) - np.min(x[:, i]))
            x[:, i] = new
        # ========================

    elif normalize == 'ZScore':
        # ====== EDIT HERE ======
        x = x.astype(np.float64)
        for i in range(x.shape[1]):
            new = (x[:, i] - np.mean(x[:, i])) / np.std(x[:, i])
            x[:, i] = new
        # =======================
    else:
        pass

    # Add 1 column for bias
    bias = np.ones((x.shape[0], 1), dtype=np.float64)
    x = np.concatenate((x, bias), axis=1)

    if shuffle:
        perm = np.random.permutation(num_data)
        x = x[perm]
        y = y[perm]

    return x, y
