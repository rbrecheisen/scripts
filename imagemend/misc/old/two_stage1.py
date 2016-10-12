__author__ = 'Ralph Brecheisen'
__date__ = "2015-07-21"
__email__ = "ralph.brecheisen@gmail.com"

# ----------------------------------------------------------------------------------------------------------------------
# In this experiment we evaluate the performance of two-stage ensemble classifier using a combination of classifiers
# trained on individual ROIs.
# ----------------------------------------------------------------------------------------------------------------------

import os
import sys

sys.path.insert(1, os.path.abspath('../..'))

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

from bin import log
from bin import create_log
from bin import add_text_to_log
from bin import get_file_text
from bin import finish_log
from bin import get_xy

# ----------------------------------------------------------------------------------------------------------------------

ROOT_DIR = '../..'
ROIS_DIR = os.path.join(ROOT_DIR, 'data/prepared/rois')
FILE_HC_SZ = os.path.join(ROIS_DIR, 'hc_sz/files.txt')
LOGS_DIR = 'logs'
OUTPUTS_DIR = 'outputs'

# ----------------------------------------------------------------------------------------------------------------------

def load_roi_names(file_name):

    roi_names = []
    f = open(file_name, 'r')
    for line in f.readlines():
        if line.startswith('#'):
            continue
        roi_names.append(line.strip().split('.')[0])
    f.close()

    return roi_names

# ----------------------------------------------------------------------------------------------------------------------

def load_roi(file_name):

    features = pd.read_csv(file_name, index_col='id')
    features.drop(['age', 'gender'], axis=1, inplace=True)
    features_HC = features[features['diagnosis1'] == 'HC']
    features_SZ = features[features['diagnosis1'] == 'SZ']

    return pd.concat([features_HC, features_SZ])

# ----------------------------------------------------------------------------------------------------------------------

def run_3():

    fold = 1
    for train, test in StratifiedKFold(subject_labels, n_folds=5):
        # Create empty table for holding predictions
        predictions = dict()
        predictions['diagnosis'] = subject_labels[train]
        # For each ROI, get voxels corresponding to training subjects
        for roi_name in roi_names:
            predictions[roi_name] = []
            X, y = get_xy(
                rois[roi_name].loc[subject_ids[train]],
                label_column='diagnosis1', exclude_columns=['diagnosis1', 'diagnosis2'])
            # Get out-of-sample predictions for each fold in the CV
            scores = []
            for train1, test1 in StratifiedKFold(subject_labels[train], n_folds=4):
                classifier = SVC()
                classifier.fit(X[train1], y[train1])
                y_pred = classifier.predict(X[test1])
                predictions[roi_name].extend(y_pred)
                scores.append(accuracy_score(y[test1], y_pred))
            print('mean score: {}'.format(np.mean(scores)))
            print('complete score: {}'.format(accuracy_score(y, predictions[roi_name])))
        # Create data frame from the predictions and save it to file
        predictions = pd.DataFrame(predictions, index=subject_ids[train])
        predictions.to_csv('outputs/roi_predictions_fold{}.txt'.format(fold))
        fold += 1
        # Now we have, for each ROI, out-of-sample predictions for all training points
        # in the initial training set. Next, fit the second model to the out-of-sample
        # predictions.
        X, y = get_xy(predictions, label_column='diagnosis', exclude_columns=['diagnosis'])
        classifier_combi = SVC(kernel='rbf')
        classifier_combi.fit(X, y)
        # Now we train a classifier on all training points of each ROI
        classifiers = {}
        for roi_name in roi_names:
            X, y = get_xy(
                rois[roi_name].loc[subject_ids[train]],
                label_column='diagnosis1', exclude_columns=['diagnosis1', 'diagnosis2'])
            classifier = SVC()
            classifier.fit(X, y)
            classifiers[roi_name] = classifier
        # Next, we apply the ROI classifiers and combined classifier to the test data
        predictions = dict()
        predictions['diagnosis'] = subject_labels[test]
        for roi_name in roi_names:
            predictions[roi_name] = []
            X, y = get_xy(
                rois[roi_name].loc[subject_ids[test]],
                label_column='diagnosis1', exclude_columns=['diagnosis1', 'diagnosis2'])
            y_pred = classifiers[roi_name].predict(X)
            predictions[roi_name].extend(y_pred)
        predictions = pd.DataFrame(predictions)
        X, y = get_xy(predictions, label_column='diagnosis', exclude_columns=['diagnosis'])
        y_pred = classifier_combi.predict(X)
        print('overall score: {}'.format(accuracy_score(y, y_pred)))

# ----------------------------------------------------------------------------------------------------------------------

def run():

    # Create log file and grab script text
    create_log()
    script_text = get_file_text('run.py')

    # Create output directory if it does not exist
    if not os.path.isdir(OUTPUTS_DIR):
        os.mkdir(OUTPUTS_DIR)

    # The code below follows a performance estimation procedure suggested by the following
    # post on Stack Overflow: https://stats.stackexchange.com/questions/102631/k-fold-cross-validation-of-ensemble-learning

    # Load ROIs
    roi_names = load_roi_names(FILE_HC_SZ)
    rois = {}
    for roi_name in roi_names:
        roi = load_roi(os.path.join(ROIS_DIR, 'hc_sz', roi_name + '_age_matched.txt'))
        for i in roi.index:
            diagnosis = roi.loc[i, 'diagnosis1']
            roi.set_value(i, 'diagnosis1', 0 if diagnosis == 'HC' else 1)
        roi['diagnosis1'] = roi['diagnosis1'].astype(int)
        rois[roi_name] = roi
        log('added ROI: {}'.format(roi_name))

    # Define parameter range for grid search later
    param_grid = [{
        'C': [2**x for x in range(-5, 15, 2)]}]

    # Get subject IDs and labels
    roi = rois[roi_names[0]]
    subject_ids = roi.index
    subject_labels = roi['diagnosis1']
    log('nr. subjects: {}'.format(len(subject_ids)))

    scores_pred = []
    scores_dist = []
    fold = 1

    # This outer CV loop is meant for averaging scores
    for train, test in StratifiedKFold(subject_labels, n_folds=10, shuffle=True):

        predictions_file = 'outputs/predictions_train{}.txt'.format(fold)
        distances_file = 'outputs/distances_train{}.txt'.format(fold)

        if not os.path.isfile(predictions_file):

            # Create empty tables for holding predictions and distances
            predictions = dict()
            predictions['diagnosis'] = subject_labels[train]
            distances = dict()
            distances['diagnosis'] = subject_labels[train]

            # Run through all ROIs
            for roi_name in roi_names:

                log('calculating out-of-sample predictions for {}'.format(roi_name))

                # Initialize prediction table for this ROI's column
                predictions[roi_name] = []
                distances[roi_name] = []

                # Get training data from the data frame
                X, y = get_xy(
                    rois[roi_name].loc[subject_ids[train]],
                    label_column='diagnosis1', exclude_columns=['diagnosis1', 'diagnosis2'])

                # Use 4-fold CV to get out-of-sample predictions for all training points
                i = 1
                for train1, test1 in StratifiedKFold(subject_labels[train], n_folds=4):

                    # Do grid search to find optimal C parameter
                    classifier = GridSearchCV(SVC(kernel='linear'), param_grid=param_grid, cv=5)
                    classifier.fit(X[train1], y[train1])

                    # Store predictions and distances for this ROI
                    y_pred = classifier.predict(X[test1])
                    predictions[roi_name].extend(y_pred)
                    y_dist = classifier.decision_function(X[test1])
                    distances[roi_name].extend(y_dist)
                    print('  step {} - {}'.format(i, 4))
                    i += 1

            # Save predictions to file
            log('saving file: {}'.format(predictions_file))
            predictions = pd.DataFrame(predictions, index=subject_ids[train])
            predictions.to_csv(predictions_file, index_label='id')

            # Save distances to file
            log('saving file: {}'.format(distances_file))
            distances = pd.DataFrame(distances, index=subject_ids[train])
            distances.to_csv(distances_file, index_label='id')

        # ---------------------

        param_grid_rbf = [{
            'C': [2**x for x in range(-5, 15, 2)],
            'gamma': [2**x for x in range(-15, 4, 2)]}]

        # Train classifier on predictions
        log('training level-2 prediction classifier')
        predictions = pd.read_csv(predictions_file, index_col='id')
        X, y = get_xy(predictions,
            label_column='diagnosis', exclude_columns=['diagnosis'])
        classifier_pred = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid_rbf, cv=5)
        classifier_pred.fit(X, y)
        log('saving level-2 prediction classifier')
        joblib.dump(classifier_pred, 'outputs/classifier_pred{}.pkl'.format(fold))

        # Train classifier on distances
        log('training level-2 distance classifier')
        distances = pd.read_csv(distances_file, index_col='id')
        X, y = get_xy(distances,
            label_column='diagnosis', exclude_columns=['diagnosis'])
        classifier_dist = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid_rbf, cv=5)
        classifier_dist.fit(X, y)
        log('saving level-2 distance classifier')
        joblib.dump(classifier_pred, 'outputs/classifier_dist{}.pkl'.format(fold))

        # ---------------------

        # Train each ROI classifier on all training points and save it to disk
        for roi_name in roi_names:

            log('training {} on all training points'.format(roi_name))

            # Skip this step if exported classifier already exists
            classifier_file = 'outputs/classifier_' + roi_name + '_train{}.pkl'.format(fold)
            if os.path.isfile(classifier_file):
                continue

            # Get training data for this fold
            X, y = get_xy(
                rois[roi_name].loc[subject_ids[train]],
                label_column='diagnosis1', exclude_columns=['diagnosis1', 'diagnosis2'])

            # Train classifier using grid search
            classifier = GridSearchCV(SVC(kernel='linear'), param_grid=param_grid, cv=5)
            classifier.fit(X, y)

            # Save best classifier to file
            log('saving {} classifier to disk'.format(roi_name))
            joblib.dump(classifier, classifier_file)

        # ---------------------

        # Load ROI classifiers from file
        classifiers = {}
        for roi_name in roi_names:
            classifier_file = 'outputs/classifier_' + roi_name + '_train{}.pkl'.format(fold)
            classifiers[roi_name] = joblib.load(classifier_file)

        # ---------------------

        predictions_test_file = 'outputs/predictions_test{}.txt'.format(fold)
        distances_test_file = 'outputs/distances_test{}.txt'.format(fold)

        if not os.path.isfile(predictions_test_file):

            predictions_test = dict()
            predictions_test['diagnosis'] = subject_labels[test]
            distances_test = dict()
            distances_test['diagnosis'] = subject_labels[test]

            for roi_name in roi_names:

                predictions_test[roi_name] = []
                distances_test[roi_name] = []

                # Get test data from the data frame
                X, y = get_xy(
                    rois[roi_name].loc[subject_ids[test]],
                    label_column='diagnosis1', exclude_columns=['diagnosis1', 'diagnosis2'])

                log('calculating predictions and distances for {}'.format(roi_name))

                # Store predictions and distances
                y_pred = classifiers[roi_name].predict(X)
                predictions_test[roi_name].extend(y_pred)
                y_dist = classifiers[roi_name].decision_function(X)
                distances_test[roi_name].extend(y_dist)

            # Save predictions to file
            log('saving predictions to file')
            predictions_test = pd.DataFrame(predictions_test, index=subject_ids[test])
            predictions_test.to_csv(predictions_test_file, index_label='id')

            # Save distances to file
            log('saving distances to file')
            distances_test = pd.DataFrame(distances_test, index=subject_ids[test])
            distances_test.to_csv(distances_test_file, index_label='id')

        # ---------------------

        # Load prediction classifier and run it on test predictions
        predictions_test = pd.read_csv(predictions_test_file, index_col='id')
        X_test, y_test = get_xy(predictions_test,
            label_column='diagnosis', exclude_columns=['diagnosis'])
        classifier_pred = joblib.load('outputs/classifier_pred{}.pkl'.format(fold))
        y_pred = classifier_pred.predict(X_test)
        scores_pred.append(accuracy_score(y_test, y_pred))
        log('score: {} (predictions)'.format(scores_pred[-1]))

        # Load distance classifier and run it on test distances
        distances_test = pd.read_csv(distances_file, index_col='id')
        X_test, y_test = get_xy(distances_test,
            label_column='diagnosis', exclude_columns=['diagnosis'])
        classifier_dist = joblib.load('outputs/classifier_dist{}.pkl'.format(fold))
        y_pred = classifier_dist.predict(X_test)
        scores_dist.append(accuracy_score(y_test, y_pred))
        log('score: {} (distances)'.format(scores_dist[-1]))

        fold += 1

    log('overall score: {} (predictions)'.format(np.mean(scores_pred)))
    log('overall score: {} (distances)'.format(np.mean(scores_dist)))

    # Append script to log and close it
    add_text_to_log(script_text)
    finish_log()

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    run()