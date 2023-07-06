import os
import numpy as np
import pandas as pd


def load_spiral(data_path):
    spiral_path = os.path.join(data_path, 'spiral')

    train = pd.read_csv(os.path.join(spiral_path,"train.csv"))
    test = pd.read_csv(os.path.join(spiral_path, "test.csv"))

    train = train.sample(frac=1).reset_index(drop=True)

    train = train.to_numpy()
    test = test.to_numpy()

    x_train = train[:,:-1]

    y_train = train[:,-1]
    x_test = test[:,:-1]
    y_test = test[:,-1]

    x1_pow = np.sin(x_train[:,0])
    x2_pow = np.cos(x_train[:,1])

    x_train = np.concatenate((
                train[:,:-1],
                x1_pow.reshape(-1,1),
                x2_pow.reshape(-1,1)
                )
                ,axis=1)


    x1_pow = np.sin(x_test[:,0])
    x2_pow = np.cos(x_test[:,1])

    x_test = np.concatenate((test[:,:-1],x1_pow.reshape(-1,1),
                x2_pow.reshape(-1,1)),axis=1)
    return x_train, y_train, x_test, y_test


def split_to_x_y(df, feature):

    input_feature = feature[:-1]
    class_label = feature[-1]

    X = df[input_feature]
    Y = df[class_label]

    return X, Y