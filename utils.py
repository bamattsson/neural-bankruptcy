import os
import shutil
import yaml

import numpy as np
import pandas as pd


def load_dataset(year, shuffle=False):
    """Loads chosen data set, mixes it and returns."""
    main_path = './data/Dane/'
    file_name = '{}year.csv'.format(year)
    file_path = os.path.join(main_path, file_name)
    df = pd.read_csv(file_path, na_values='?')
    Y = df['class'].values
    X = df.drop('class', axis=1).values
    if shuffle:
        shuffled_idx = np.random.permutation(len(Y))
        X = X[shuffled_idx, :]
        Y = Y[shuffled_idx]
    return X, Y


def load_yaml_and_save(yaml_path, run_path):
    with open(yaml_path, 'r') as f:
        config = yaml.load(f)
    save_path = os.path.join(run_path, 'cfg.yml')
    shutil.copyfile(yaml_path, save_path)
    return config


def split_dataset(X, Y, share_last):
    """
    Splits data set into two parts.

    Args:
        X (np.array): X data
        Y (np.array): Y data
        share_last (float): how large share of the dataset that should be in
            the last part of the data set
    """
    split_point = int(len(Y) * share_last)
    X_last = X[:split_point, :]
    X_first = X[split_point:, :]
    Y_last = Y[:split_point]
    Y_first = Y[split_point:]
    return X_first, Y_first, X_last, Y_last
