import os
import shutil
import time
import numpy as np
import pandas as pd
import yaml


def load_yaml_and_save(yaml_path, run_path):
    with open(yaml_path, 'r') as f:
        config = yaml.load(f)
    save_path = os.path.join(run_path, 'cfg.yml')
    shutil.copyfile(yaml_path, save_path)
    return config


def load_dataset(year, shuffle=False, seed=42):
    """Loads chosen data set, mixes it and returns."""
    main_path = './data/Dane/'
    file_name = '{}year.csv'.format(year)
    file_path = os.path.join(main_path, file_name)
    df = pd.read_csv(file_path)
    Y = df['class'].values
    X = df.drop('class', axis=1).values
    if shuffle:
        raise NotImplementedError
    return X, Y


def perform_one_run():
    # takes data set
    # trains model
    # evaluates experiment
    # returns results
    pass


def main(yaml_path='./config.yml', run_name=None):

    if run_name is None:
        run_name = time.strftime('%Y%m%d-%H%M', time.localtime())
    run_path = os.path.join('./output', run_name)
    if not os.path.exists(run_path):
        os.makedirs(run_path)

    config = load_yaml_and_save(yaml_path, run_path)


if __name__ == '__main__':
    main()
