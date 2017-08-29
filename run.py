import sys
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, model_selection
from utils import split_dataset, load_dataset, load_yaml_and_save
from algorithms import (RandomGuessAlgorithm, RandomForestAlgorithm,
        GradientBoostingAlgorithm, MultilayerPerceptron)
from data_processors import Imputer, Processor


def main(yaml_path='./config.yml', run_name=None):

    # Create output directory where experiment is saved
    if run_name is None:
        run_name = time.strftime('%Y%m%d-%H%M', time.localtime())
    run_path = os.path.join('./output', run_name)
    if not os.path.exists(run_path):
        os.makedirs(run_path)

    config = load_yaml_and_save(yaml_path, run_path)

    # Do the specified experiments
    np.random.seed(config['experiment']['np_random_seed'])
    for year in config['experiment']['years']:
        do_experiment_for_one_year(run_path, year, config)
    # Show figures if any have been generated
    if config['analysis']['plot_roc']:
        plt.show()


def do_experiment_for_one_year(run_path, year, config):
    """Performs the specified experiments for one year."""
    X, Y = load_dataset(year, shuffle=config['experiment']['shuffle_data'])
    if config['experiment']['type'] == 'single':
        X_train, Y_train, X_test, Y_test = split_dataset(X, Y,
                config['experiment']['test_share'])
        results = perform_one_experiment(X_train, Y_train, X_test, Y_test,
                config)
    elif config['experiment']['type'] == 'cv':
        results = perform_cv_runs(X, Y, config)

    results_path = os.path.join(run_path, 'results_year{}.pkl'.format(year))
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    show_results(results, year, **config['analysis'])


def perform_one_experiment(X_train, Y_train, X_test, Y_test, config):
    """Performs one experiment with a given data set and generates results."""
    # Prepare data
    processor = Processor(**config['processor_params'])
    X_train = processor.fit_transform(X_train)
    X_test = processor.transform(X_test)
    imputer = Imputer(**config['imputer_params'])
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Creates the algorithm object
    algorithm_name = config['experiment']['algorithm']
    if algorithm_name == 'random_guess':
        algorithm = RandomGuessAlgorithm(**config['algo_params'])
    elif algorithm_name == 'rf':
        algorithm = RandomForestAlgorithm(**config['algo_params'])
    elif algorithm_name == 'multilayer_perceptron':
        algorithm = MultilayerPerceptron(n_input=X_train.shape[1],
                **config['algo_params'])
    elif algorithm_name == 'gradient_boosting':
        algorithm = GradientBoostingAlgorithm(**config['algo_params'])
    else:
        raise NotImplementedError('Algorithm {} is not an available option'
                .format(algorithm_name))

    # Perform experiment
    results = dict()
    results['fit_info'] = algorithm.fit(X_train, Y_train)
    pred_proba = algorithm.predict_proba(X_test)
    pred = np.argmax(pred_proba, axis=1)

    # Calculate and save results
    results['log_loss'] = metrics.log_loss(Y_test, pred_proba[:, 1])
    results['accuracy'] = metrics.accuracy_score(Y_test, pred)
    results['recall'] = metrics.recall_score(Y_test, pred, labels=[0, 1])
    results['precision'] = metrics.precision_score(Y_test, pred, labels=[0, 1])
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, pred_proba[:, 1])
    results['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
    results['roc_auc'] = metrics.auc(fpr, tpr)
    results['classification_report'] = metrics.classification_report(Y_test,
            pred, labels=[0, 1])

    return results


def perform_cv_runs(X, Y, config):
    """Performs cv test for one year."""
    # split up in cv and perform runs
    cv_splitter = model_selection.KFold(n_splits=config['experiment']
            ['n_folds'])
    result_list = []
    for train_index, test_index in cv_splitter.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        results_one_experiment = perform_one_experiment(X_train, Y_train,
                X_test, Y_test, config)
        result_list.append(results_one_experiment)

    # Loop through all the generated results and save to one place
    results = dict()
    for metric in result_list[0]:
        # Roc curve comes in a dict and are treated seperately
        if type(result_list[0][metric]) == dict:
            results[metric] = dict()
            for submetric in result_list[0][metric]:
                arrays = [result_from_cv[metric][submetric] for result_from_cv
                        in result_list]
                results[metric][submetric] = arrays
        # Float values are saved as a list of all values but also as mean and std
        elif isinstance(result_list[0][metric], float):
            results[metric] = dict()
            results[metric]['values'] = np.array([result_from_cv[metric] for
                result_from_cv in result_list])
            results[metric]['mean'] = results[metric]['values'].mean()
            results[metric]['std'] = results[metric]['values'].std()
        # Other values are saved in lists
        else:
            value = [result_from_cv[metric] for result_from_cv in result_list]
            results[metric] = value

    return results


def show_results(results, year, print_results=[], plot_roc=False):
    """Print and plot the results that have been generated."""
    # Print results
    if (len(print_results) > 0):
        print('\nResults for year {}:'.format(year))
        for metric in print_results:
            if type(results[metric]) == dict:
                print('{}={:.2f}, std={:.2f}'.format(metric,
                    results[metric]['mean'], results[metric]['std']))
            elif type(results[metric]) == str:
                print(results[metric])
            elif isinstance(results[metric], float):
                print('{}={:.2f}'.format(metric, results[metric]))
            else:
                print('{} cannot be printed.'.format(metric))

    # Plot results
    if plot_roc:
        plt.figure(year)
        plt.title('roc curve year {}'.format(year))
        plt.plot((0, 1), (0, 1), ls='--', c='k')
        if type(results['roc_curve']['fpr']) == list:
            # A CV run with multiple arrays
            for fpr, tpr in zip(results['roc_curve']['fpr'],
                    results['roc_curve']['tpr']):
                plt.plot(fpr, tpr)
        else:
            # Not a CV run
            plt.plot(results['roc_curve']['fpr'], results['roc_curve']['tpr'])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')


if __name__ == '__main__':
    try:
        yaml_path = sys.argv[1]
    except IndexError as e:
        print('You have to specify the config.yaml to use as `python run.py '
            'example_config.yaml`')
        print('Exiting.')
        sys.exit()
    main(yaml_path=yaml_path)
