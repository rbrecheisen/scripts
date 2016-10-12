__author__ = 'Ralph Brecheisen'
__date__ = "2015-07-28"
__email__ = "ralph.brecheisen@gmail.com"

# ----------------------------------------------------------------------------------------------------------------------

import os
import sys
import math

sys.path.insert(1, os.path.abspath('../..'))

import numpy as np
import pandas as pd
import multiprocessing
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

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

def load_roi(roi_name):

    file_name = os.path.join(ROIS_DIR, 'hc_sz', roi_name + '_age_matched.txt')
    roi = pd.read_csv(file_name, index_col='id')
    for i in roi.index:
        diagnosis = roi.loc[i, 'diagnosis1']
        roi.set_value(i, 'diagnosis1', -1 if diagnosis == 'HC' else 1)
    roi['diagnosis1'] = roi['diagnosis1'].astype(int)
    return roi

# ----------------------------------------------------------------------------------------------------------------------

def load_rois(rois):

    roi_names = load_roi_names(FILE_HC_SZ)
    for roi_name in roi_names:
        print('loading {}'.format(roi_name))
        rois[roi_name] = load_roi(roi_name)

# ----------------------------------------------------------------------------------------------------------------------

def get_subject_ids_and_labels(roi):

    return roi.index, roi['diagnosis1']

# ----------------------------------------------------------------------------------------------------------------------

def train_classifier(rois, roi_name, subject_ids, queue, probability):

    X, y = get_xy(
        rois[roi_name].loc[subject_ids],
        label_column='diagnosis1',
        exclude_columns=['age', 'gender', 'diagnosis1', 'diagnosis2'])
    param_grid = [{
        'C': [2**x for x in range(-5, 15, 2)]}]
    scores = []
    for train, test in StratifiedKFold(y, n_folds=10, shuffle=True):
        classifier = GridSearchCV(SVC(kernel='linear', probability=probability), param_grid=param_grid)
        classifier.fit(X[train], y[train])
        score = accuracy_score(y[test], classifier.predict(X[test]))
        scores.append(score)
    queue.put((roi_name, np.mean(scores)))

# ----------------------------------------------------------------------------------------------------------------------

def save_classifier(rois, roi_name, subject_ids, probability):

    X, y = get_xy(
        rois[roi_name].loc[subject_ids],
        label_column='diagnosis1',
        exclude_columns=['age', 'gender', 'diagnosis1', 'diagnosis2'])
    param_grid = [{
        'C': [2**x for x in range(-5, 15, 2)]}]
    classifier = GridSearchCV(SVC(kernel='linear', probability=probability), param_grid=param_grid)
    classifier.fit(X, y)
    if not os.path.isdir('outputs/' + roi_name):
        os.mkdir('outputs/' + roi_name)
    joblib.dump(classifier, 'outputs/' + roi_name + '/classifier.pkl')

# ----------------------------------------------------------------------------------------------------------------------

def load_classifier(roi_name):

    return joblib.load('outputs/' + roi_name + '/classifier.pkl')

# ----------------------------------------------------------------------------------------------------------------------

def get_predictions(rois, roi_name, subject_ids, classifier):

    X, y = get_xy(
        rois[roi_name].loc[subject_ids],
        label_column='diagnosis1',
        exclude_columns=['age', 'gender', 'diagnosis1', 'diagnosis2'])
    return np.array(classifier.predict(X))

# ----------------------------------------------------------------------------------------------------------------------

def get_probabilities(rois, roi_name, subject_ids, classifier):

    X, y = get_xy(
        rois[roi_name].loc[subject_ids],
        label_column='diagnosis1',
        exclude_columns=['age', 'gender', 'diagnosis1', 'diagnosis2'])
    probabilities = classifier.predict_proba(X)
    probabilities = np.transpose(probabilities)
    return probabilities[0], probabilities[1]

# ----------------------------------------------------------------------------------------------------------------------

def train_classifiers(rois, subject_ids, probability=False):

    print('training classifiers')
    queue = multiprocessing.Queue()
    threads = []
    for roi_name in rois.keys():
        thread = multiprocessing.Process(
            target=train_classifier, args=(rois, roi_name, subject_ids, queue, probability))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

    scores = {}
    while not queue.empty():
        roi_name, score = queue.get()
        scores[roi_name] = score

    print('saving classifiers')
    threads = []
    for roi_name in rois.keys():
        thread = multiprocessing.Process(
            target=save_classifier, args=(rois, roi_name, subject_ids, probability))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

    classifiers = {}
    for roi_name in rois.keys():
        classifiers[roi_name] = load_classifier(roi_name)

    return classifiers, scores

# ----------------------------------------------------------------------------------------------------------------------

def combine_predictions(rois, subject_ids, classifiers, scores):

    predictions = {}
    for roi_name in classifiers.keys():
        predictions[roi_name] = get_predictions(rois, roi_name, subject_ids, classifiers[roi_name])
        predictions[roi_name] = scores[roi_name] * predictions[roi_name]
    predictions = pd.DataFrame(predictions, index=subject_ids)

    y = []
    for subject_id in subject_ids:
        y.append(-1 if np.sum(predictions.loc[subject_id]) < 0 else 1)
    return np.array(y)

# ----------------------------------------------------------------------------------------------------------------------

def combine_probabilities(rois, subject_ids, classifiers, scores):

    probabilities1 = {}
    probabilities2 = {}
    for roi_name in classifiers.keys():
        tmp1, tmp2 = get_probabilities(rois, roi_name, subject_ids, classifiers[roi_name])
        probabilities1[roi_name] = tmp1
        probabilities2[roi_name] = tmp2

    probabilities1 = pd.DataFrame(probabilities1, index=subject_ids)
    probabilities2 = pd.DataFrame(probabilities2, index=subject_ids)

    total_score = 0
    for roi_name in scores.keys():
        total_score = scores[roi_name]
    total_score /= len(scores)

    y = []
    for subject_id in subject_ids:
        p1 = 0
        p2 = 0
        for roi_name in classifiers.keys():
            score = scores[roi_name] / total_score
            p1 += score * math.log(probabilities1.loc[subject_id, roi_name])
            p2 += score * math.log(probabilities2.loc[subject_id, roi_name])
        y.append(-1 if p1 > p2 else 1)

    return np.array(y)

# ----------------------------------------------------------------------------------------------------------------------

def run():

    if not os.path.isdir('outputs'):
        os.mkdir('outputs')

    manager = multiprocessing.Manager()
    rois = manager.dict()
    load_rois(rois)

    subject_ids, subject_labels = get_subject_ids_and_labels(rois.values()[0])

    pred_scores = []
    for train, test in StratifiedKFold(subject_labels, n_folds=10, shuffle=True):
        classifiers, scores = train_classifiers(rois, subject_ids[train])
        y_pred = combine_predictions(
            rois, subject_ids[test], classifiers, scores)
        pred_scores.append(accuracy_score(subject_labels[test], y_pred))
        print('score: {}'.format(pred_scores[-1]))

    prob_scores = []
    for train, test in StratifiedKFold(subject_labels, n_folds=10, shuffle=True):
        classifiers, scores = train_classifiers(rois, subject_ids[train])
        y_pred = combine_probabilities(
            rois, subject_ids[test], classifiers, scores)
        prob_scores.append(accuracy_score(subject_labels[test], y_pred))
        print('score: {}'.format(prob_scores[-1]))

    print('final score (predictions)  : {}'.format(np.mean(pred_scores)))
    print('final score (probabilities): {}'.format(np.mean(prob_scores)))

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    run()