# ----------------------------------------------------------------------------------------------------------------------
# This script evaluates performance of a number of classifiers.
# ----------------------------------------------------------------------------------------------------------------------
__author__ = 'Ralph Brecheisen'

import os
import sys

sys.path.insert(1, os.path.abspath('../..'))

import gen_report

from scripts import util
from scripts import const
from scripts import logging
from scripts import prepare
from scripts import evaluate

import numpy as np
import multiprocessing

import GPy

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

LOG = logging.Logger()


# ----------------------------------------------------------------------------------------------------------------------
def create_info(train_score, test, y_pred, y_true, best_params):

    return {
        'train_score': train_score,
        'ids':  list(test),
        'pred': list(y_pred),
        'true': list(y_true),
        'best_params': best_params,
    }


# ----------------------------------------------------------------------------------------------------------------------
def run_svm_rbf(X, y, train, test, key, k):

    param_grid = [{
        'C':     [2**x for x in range(-5, 15, 2)],
        'gamma': [2**x for x in range(-15, 4, 2)],
    }]

    classifier = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, scoring='accuracy')
    classifier.fit(X[train], y[train])
    y_pred = classifier.predict(X[test])
    y_true = y[test]

    train_score = evaluate.get_accuracy_score(y[train], classifier.predict(X[train]))

    info = create_info(train_score, test, y_pred, y_true, {
        'C': classifier.best_params_['C'],
        'gamma': classifier.best_params_['gamma']
    })

    file_name = 'tmp/{}-{}.tmp'.format(key, str(k))
    util.to_file(file_name, info)


# ----------------------------------------------------------------------------------------------------------------------
def run_svm_poly(X, y, train, test, key, k):

    param_grid = [{
        'C':      [2**x for x in range(-5, 15, 2)],
        'gamma':  [2**x for x in range(-15, 4, 2)],
        'degree': [1, 2, 3, 4, 5],
        'coef0':  [-1, 0, 1],
    }]

    classifier = GridSearchCV(SVC(kernel='poly'), param_grid=param_grid, scoring='accuracy')
    classifier.fit(X[train], y[train])
    y_pred = classifier.predict(X[test])
    y_true = y[test]

    train_score = evaluate.get_accuracy_score(y[train], classifier.predict(X[train]))

    info = create_info(train_score, test, y_pred, y_true, {
        'C': classifier.best_params_['C'],
        'gamma': classifier.best_params_['gamma'],
        'degree': classifier.best_params_['degree'],
        'coef0': classifier.best_params_['coef0'],
    })

    file_name = 'tmp/{}-{}.tmp'.format(key, str(k))
    util.to_file(file_name, info)


# ----------------------------------------------------------------------------------------------------------------------
def run_svm_sigmoid(X, y, train, test, key, k):

    param_grid = [{
        'C':     [2**x for x in range(-5, 15, 2)],
        'gamma': [2**x for x in range(-15, 4, 2)],
        'coef0': [-1, 0, 1],
    }]

    classifier = GridSearchCV(SVC(kernel='sigmoid'), param_grid=param_grid, scoring='accuracy')
    classifier.fit(X[train], y[train])
    y_pred = classifier.predict(X[test])
    y_true = y[test]

    train_score = evaluate.get_accuracy_score(y[train], classifier.predict(X[train]))

    info = create_info(train_score, test, y_pred, y_true, {
        'C': classifier.best_params_['C'],
        'gamma': classifier.best_params_['gamma'],
        'coef0': classifier.best_params_['coef0'],
    })

    file_name = 'tmp/{}-{}.tmp'.format(key, str(k))
    util.to_file(file_name, info)


# ----------------------------------------------------------------------------------------------------------------------
def run_random_forest(X, y, train, test, key, k):

    param_grid = [{
        'n_estimators': [1000],
        'criterion': ['gini', 'entropy'],
        'max_features': np.linspace(0.01, 0.5, num=10)
    }]

    classifier = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=3, refit=True)
    classifier.fit(X[train], y[train])
    y_pred = classifier.predict(X[test])
    y_true = y[test]

    train_score = evaluate.get_accuracy_score(y[train], classifier.predict(X[train]))

    info = create_info(train_score, test, y_pred, y_true, {
        'n_estimators': classifier.best_params_['n_estimators'],
        'criterion': classifier.best_params_['criterion'],
        'max_features': classifier.best_params_['max_features'],
    })

    file_name = 'tmp/{}-{}.tmp'.format(key, str(k))
    util.to_file(file_name, info)


# ----------------------------------------------------------------------------------------------------------------------
def run_gp(X, y, train, test, key, k):

    X_train = X[train]
    Y_train = np.array([y[train]]).T
    classifier = GPy.models.GPClassification(
        X_train, Y_train, GPy.kern.RBF(X_train.shape[1], lengthscale=4.0))
    for _ in range(5):
        classifier.optimize()

    X_test = X[test]
    Y_test = np.array([y[test]]).T
    probs = classifier.predict(X_test)[0].T[0]
    y_pred = np.ones(len(y[test]))
    y_pred[probs<0.5] = 0

    train_error_rate, _, _, _, _ = GPy.util.classification.conf_matrix(classifier.predict(X_train)[0], Y_train, show=False)
    train_score = 1.0 - train_error_rate

    info = create_info(train_score, test, y_pred, y[test], {})

    file_name = 'tmp/{}-{}.tmp'.format(key, str(k))
    util.to_file(file_name, info)


# ----------------------------------------------------------------------------------------------------------------------
def run_parallel(classifiers, features, n_iters=10):

    X, y = util.get_xy(
        features,
        target_column='diagnosis',
        exclude_columns=['age', 'gender', 'diagnosis'])

    if not os.path.isdir('tmp'):
        os.mkdir('tmp')

    threads = []
    k = 0
    for i in range(n_iters):
        for train, test in StratifiedKFold(y, n_folds=10, shuffle=True):
            for key in classifiers.keys():
                thread = multiprocessing.Process(
                    target=classifiers[key], args=(X, y, train, test, key, k))
                threads.append(thread)
                thread.start()
            k += 1
    for thread in threads:
        thread.join()

    perf_info = {}
    for key in classifiers.keys():
        for i in range(k):
            if key not in perf_info:
                perf_info[key] = []
            info = util.from_file('tmp/{}-{}.tmp'.format(key, str(i)))
            perf_info[key].append(info)

    util.remove_files('tmp/*.tmp')

    return perf_info


# ----------------------------------------------------------------------------------------------------------------------
def run_sequential(classifiers, features, n_iters=10):

    X, y = util.get_xy(
        features,
        target_column='diagnosis',
        exclude_columns=['age', 'gender', 'diagnosis'])

    if not os.path.isdir('tmp'):
        os.mkdir('tmp')

    k = 0
    for i in range(n_iters):
        for train, test in StratifiedKFold(y, n_folds=10, shuffle=True):
            for key in classifiers.keys():
                classifiers[key](X, y, train, test, key, k)
            k += 1

    perf_info = {}
    for key in classifiers.keys():
        for i in range(k):
            if key not in perf_info:
                perf_info[key] = []
            info = util.from_file('tmp/{}-{}.tmp'.format(key, str(i)))
            perf_info[key].append(info)

    util.remove_files('tmp/*.tmp')
    return perf_info


# ----------------------------------------------------------------------------------------------------------------------
def run():

    # Load FreeSurfer features
    features = util.load_features(const.FREESURFER_FILE)
    # Perform age-matching between healthy and schizophrenia patients
    features = prepare.match_ages(features, 'HC', 'SZ', 2, nr_labels=2)
    # Remove constant features
    features = prepare.remove_constant_features(features)
    # Normalize numerical features across subjects (excluding 'age')
    features = prepare.normalize_across_subjects(features, exclude=['age'])
    # Subtract age and gender confounds
    features = prepare.residualize(features, ['age', 'gender', 'EstimatedTotalIntraCranialVol'])
    # Replace HC and SZ labels with -1 and 1
    features['diagnosis'].replace(['HC', 'SZ'], [1, 0], inplace=True)

    # Create a list of classifier functions to run in parallel. Each function runs
    # the given classifier and writes its performance output to a randomly named
    # file. The file name is added to the queue so we can collect the content of
    # these files later to produce the final output result
    classifiers = {
        'svm-rbf': run_svm_rbf,
        'svm-poly': run_svm_poly,
        'svm-sigmoid': run_svm_sigmoid,
        'random-forest': run_random_forest,
        'gp': run_gp,
    }

    # Run all classifiers in parallel
    perf_info = run_parallel(classifiers, features, 10)
    # perf_info = run_sequential(classifiers, features, 1)

    # Save performance info to JSON
    if not os.path.isdir(const.RESULTS_DIR):
        os.mkdir(const.RESULTS_DIR)
    results_file = os.path.join(const.RESULTS_DIR, '{}_perf_info.json'.format(LOG.get_timestamp()))
    util.to_file(results_file, perf_info)

    # Close log file
    LOG.close()

    # Generate report
    gen_report.run(results_file)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    run()