# ------------------------------------------------------------------------------
# This script runs an SVM on each ROI and creates a ranked list of ROIs according
# to their predictive power. Permutation testing will be done to establish whether
# the classification scores of each ROI are significantly better than chance.
# ------------------------------------------------------------------------------

__author__ = 'Ralph Brecheisen'
__date__ = '2015-07-08'
__email__ = 'ralph.brecheisen@gmail.com'

# ------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer

sys.path.insert(1, os.path.abspath('../..'))

from bin import log
from bin import create_log
from bin import add_text_to_log
from bin import finish_log
from bin import get_log_timestamp
from bin import get_file_text
from bin import get_xy
from bin import get_accuracy
from bin import write_dict_to_json

# ------------------------------------------------------------------------------

ROOT_DIR = '../..'
ROI_ROOT_DIR = ROOT_DIR + '/data/prepared/rois'
ROI_FILE_HC_SZ = ROI_ROOT_DIR + '/hc_sz/files.txt'
LOGS_DIR = 'logs'
OUTPUTS_DIR = 'outputs'

# ------------------------------------------------------------------------------

def run_svm(X, y, score_func, kernel, nr_folds=10):
    """
    Runs an optimized SVM using a grid-search on the C parameter.
    :param X: Input features
    :param y: True labels
    :param score_func: Score function
    :param nr_folds: Number of CV folds
    :return: Performance scores, selected kernels, C parameters, gammas
    """

    # Setup parameter ranges for grid search
    param_grid = [{
        'C':      [2**x for x in range(-5, 15, 2)],
        'gamma':  [2**x for x in range(-15, 4, 2)] if kernel == 'rbf' else [0.0],
        'kernel': [kernel]}]

    scores = []
    gammas = []
    cs = []

    for j in range(10):

        for i, (train, test) in enumerate(StratifiedKFold(y, n_folds=nr_folds, shuffle=True)):

            # Define SVM classifier with grid-based parameter search
            classifier = GridSearchCV(
                SVC(), param_grid=param_grid, scoring=make_scorer(score_func), cv=5)
            classifier.fit(X[train], y[train])
            score = score_func(y[test], classifier.predict(X[test]))

            # Get optimized hyper parameters
            gamma = classifier.best_params_['gamma']
            if kernel == 'linear':
                gamma = 0.0
            c = classifier.best_params_['C']

            # Add parameters to respective lists
            scores.append(score)
            cs.append(c)
            if kernel == 'rbf':
                gammas.append(gamma)

            # Report score and best parameters for this iteration
            log('  score: {} (kernel = {}, C = {}, gamma = {}'.format(score, kernel, c, gamma))

    return scores, cs, gammas

# ------------------------------------------------------------------------------

def run_permutation_test(X, y, score_func, nr_folds=10, nr_permutations=100):
    """
    Runs permutation test on given classifier using score_func.
    :param X: Input features
    :param y: Labels
    :param score_func: Score function
    :param nr_folds: Number of CV folds
    :param nr_permutations: Number of permutations
    :return: Estimated p-value
    """

    # First calculate scores on original data
    scores = run_svm(X, y, score_func, nr_folds)
    log('average score: {}'.format(np.mean(scores)))

    # Next, do permutation of labels N times
    param_grid = [{
        'C': [2**x for x in range(-5, 15, 2)]}]
    permuted_scores = []

    for i in range(nr_permutations):

        # Permute the labels
        y_perm = np.random.permutation(y)

        # Create train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_perm, test_size=0.25)

        # Train SVM using grid-search to optimize C parameter
        classifier = GridSearchCV(
            SVC(), param_grid=param_grid, scoring=make_scorer(score_func))
        classifier.fit(X_train, y_train)

        # Calculate score based on score function
        score = score_func(y_test, classifier.predict(X_test))
        permuted_scores.append(score)

    log('average permuted score: {}'.format(np.mean(permuted_scores)))

    # Calculate p-value by averaging over scores calculated on
    # the original data (using CV)
    p_values = []
    for score in scores:
        count = 0
        for permuted_score in permuted_scores:
            if permuted_score >= score:
                count += 1
        p_values.append((count + 1) / (nr_permutations + 1))
    p_value = np.mean(p_values)

    log('p-value: {}'.format(p_value))

    return scores, permuted_scores, p_value

# ------------------------------------------------------------------------------

def init_output_dict(roi_names):
    """
    Creates empty output dictionary to store information such as
    performance scores.
    :return: Output dictionary
    """
    roi_info = {}

    for roi_name in roi_names:

        roi_info[roi_name] = {}

        roi_info[roi_name]['all'] = {}
        roi_info[roi_name]['all']['linear'] = {}
        roi_info[roi_name]['all']['linear']['accuracy'] = {}

        roi_info[roi_name]['males'] = {}
        roi_info[roi_name]['males']['linear'] = {}
        roi_info[roi_name]['males']['linear']['accuracy'] = {}

        roi_info[roi_name]['females'] = {}
        roi_info[roi_name]['females']['linear'] = {}
        roi_info[roi_name]['females']['linear']['accuracy'] = {}

    return roi_info

# ------------------------------------------------------------------------------

def run():
    """
    This script runs both linear and nonlinear SVMs on each ROI and creates
    a list ranking the ROIs ordered by classification power.
    :return: None
    """

    # Create new log file
    create_log(subdir=LOGS_DIR)

    # Grab script text for inclusion in the log file at the end
    script_text = get_file_text('run.py')

    # Load ROI file names
    roi_names = []
    with open(ROI_FILE_HC_SZ, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            roi_name = line.strip().split('.')[0]
            roi_names.append(roi_name)

    # Initialize dictionary with ROI info like performance scores
    roi_info = init_output_dict(roi_names)

    # Run through each ROI file name
    for roi_name in roi_names:

        # BOTH GENDERS

        roi_file_age_matched = roi_name + '_age_matched.txt'
        log('loading {}'.format(roi_file_age_matched))
        features = pd.read_csv(ROI_ROOT_DIR + '/hc_sz/' + roi_file_age_matched, index_col='id')

        # Make sure to select only schizophrenia patients and controls
        features_HC = features[features['diagnosis1'] == 'HC']
        features_SZ = features[features['diagnosis1'] == 'SZ']
        features = pd.concat([features_HC, features_SZ])

        # Get X, y for both genders
        X, y = get_xy(
            features=features,
            label_column='diagnosis1',
            exclude_columns=['diagnosis1', 'diagnosis2', 'age', 'gender'])

        # Run SVM
        scores, cs, gammas = run_svm(X, y, get_accuracy, kernel='linear')
        log('accuracy: {} +/- {}'.format(np.mean(scores), np.std(scores)))
        roi_info[roi_name]['all']['linear']['accuracy']['mean'] = np.mean(scores)
        roi_info[roi_name]['all']['linear']['accuracy']['stddev'] = np.std(scores)
        roi_info[roi_name]['all']['linear']['cs'] = cs

        # MALES

        roi_file_M_age_matched = roi_name + '_male_age_matched.txt'
        log('loading {}'.format(roi_file_M_age_matched))
        features = pd.read_csv(ROI_ROOT_DIR + '/hc_sz/' + roi_file_M_age_matched, index_col='id')

        # Select only schizophrenia patients and controls
        features_HC = features[features['diagnosis1'] == 'HC']
        features_SZ = features[features['diagnosis1'] == 'SZ']
        features = pd.concat([features_HC, features_SZ])

        # Get X, y for males
        X, y = get_xy(
            features=features,
            label_column='diagnosis1',
            exclude_columns=['diagnosis1', 'diagnosis2', 'age', 'gender'])

        # Run linear SVM
        scores, cs, gammas = run_svm(X, y, get_accuracy, kernel='linear')
        log('accuracy: {} +/- {} (linear)'.format(np.mean(scores), np.std(scores)))
        roi_info[roi_name]['males']['linear']['accuracy']['mean'] = np.mean(scores)
        roi_info[roi_name]['males']['linear']['accuracy']['stddev'] = np.std(scores)
        roi_info[roi_name]['males']['linear']['cs'] = cs

        # FEMALES

        roi_file_F_age_matched = roi_name + '_female_age_matched.txt'
        log('loading {}'.format(roi_file_F_age_matched))
        features = pd.read_csv(ROI_ROOT_DIR + '/hc_sz/' + roi_file_F_age_matched, index_col='id')

        # Select only schizophrenia patients and controls
        features_HC = features[features['diagnosis1'] == 'HC']
        features_SZ = features[features['diagnosis1'] == 'SZ']
        features = pd.concat([features_HC, features_SZ])

        # Get X, y for females
        X, y = get_xy(
            features=features,
            label_column='diagnosis1',
            exclude_columns=['diagnosis1', 'diagnosis2', 'age', 'gender'])

        # Run linear SVM
        scores, cs, gammas = run_svm(X, y, get_accuracy, kernel='linear')
        log('accuracy: {} +/- {} (linear)'.format(np.mean(scores), np.std(scores)))
        roi_info[roi_name]['females']['linear']['accuracy']['mean'] = np.mean(scores)
        roi_info[roi_name]['females']['linear']['accuracy']['stddev'] = np.std(scores)
        roi_info[roi_name]['females']['linear']['cs'] = cs

    # Write scores to JSON
    if not os.path.isdir(OUTPUTS_DIR):
        os.mkdir(OUTPUTS_DIR)
    output_file = os.path.join(OUTPUTS_DIR, get_log_timestamp() + '_scores.json')
    write_dict_to_json(output_file, roi_info)

    # Append this script's text to the log file for backup
    add_text_to_log(script_text)
    finish_log()

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    run()