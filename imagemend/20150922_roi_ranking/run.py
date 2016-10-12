# ----------------------------------------------------------------------------------------------------------------------
# This script uses Gaussian Process classifiers to rank ROIs.
# ----------------------------------------------------------------------------------------------------------------------
__author__ = 'Ralph Brecheisen'

import os
import sys
import glob
import json

sys.path.insert(1, os.path.abspath('../..'))

from scripts import util
from scripts import const
from scripts import logging
from scripts import prepare
from scripts import timing

import GPy
import multiprocessing
import numpy as np
import pandas as pd

from sklearn.cross_validation import StratifiedKFold


# ----------------------------------------------------------------------------------------------------------------------
def load_roi(file_name, label1, label2):

	roi_data = util.load_features_for(file_name, [label1, label2])
	meta_file = os.path.join(const.DATA_DIR, 'meta.txt')
	roi_data = prepare.add_feature_columns(roi_data, ['age', 'diagnosis'], meta_file)
	roi_data = prepare.match_ages(roi_data, label1, label2, 2, nr_labels=2)

	return roi_data


# ----------------------------------------------------------------------------------------------------------------------
def run_gp(roi_name, i, kernels, n_iters):

	# Specify meta file with additional features
	meta_file = os.path.join(const.DATA_DIR, 'meta.txt')
	# Specify ROI file name to load
	roi_file = os.path.join(const.ROI_DIR, roi_name + '_s0.txt')
	# Load ROI voxel intensities
	features = util.load_features(roi_file)
	# Add columns for age, gender and diagnosis
	features = prepare.add_feature_columns(features, ['age', 'gender', 'diagnosis'], meta_file)
	# Match ages between HC and SZ patients
	features = prepare.match_ages(features, 'HC', 'SZ', 2, nr_labels=2)
	# Subtract age and gender confounds
	features = prepare.residualize(features, ['age', 'gender'], verbose=False)
	# Replace HC and SZ labels with -1 and 1
	features['diagnosis'].replace(['HC', 'SZ'], [1, 0], inplace=True)
	# Get number of features
	nr_features = features.shape[1]

	# Load subject ID sets for this permutation
	i = str(i)
	with open('ids.json', 'r') as f:
		subject_ids = json.load(f)[i]

	# Convert data to NumPy arrays
	X, y = util.get_xy(
		features,
		target_column='diagnosis',
		exclude_columns=['age', 'gender', 'diagnosis'])

	scores = {}

	for knl in kernels:

		scores[knl] = []

		for j in range(n_iters):

			j = str(j)
			k_range = len(subject_ids[j].keys())

			for k in range(k_range):

				k = str(k)
				test = subject_ids[j][k]['test']
				train = subject_ids[j][k]['train']

				X_train = X[train]
				Y_train = np.array([y[train]]).T
				kernel = util.build_gp_kernel(knl, X_train.shape[1])
				classifier = GPy.models.GPClassification(X_train, Y_train, kernel)
				for _ in range(5):
					classifier.optimize()

				X_test = X[test]
				Y_test = np.array([y[test]]).T
				probs = classifier.predict(X_test)[0]
				error_rate, _, _, _, _ = GPy.util.classification.conf_matrix(probs, Y_test, show=False)
				scores[knl].append(1.0 - error_rate)

	# Determine file write mode
	score_file = os.path.join(const.RESULTS_DIR, roi_name + '_scores.txt')
	write_mode = 'w'
	if os.path.isfile(score_file):
		write_mode = 'a'

	# Write output to file
	with open(score_file, write_mode) as f:
		if write_mode == 'w':
			header = list()
			header.append('roi_name')
			header.append('nr_features')
			for knl in kernels:
				header.append(knl)
			f.write(','.join(header) + '\n')
		line = list()
		line.append(roi_name)
		line.append(str(nr_features))
		for knl in kernels:
			line.append(str(np.mean(scores[knl])))
		f.write(','.join(line) + '\n')

# ----------------------------------------------------------------------------------------------------------------------
def run():

	# Create results output directory. If it contains ROI score files
	# delete all of them.
	if not os.path.isdir(const.RESULTS_DIR):
		os.mkdir(const.RESULTS_DIR)
	for f in glob.glob(os.path.join(const.RESULTS_DIR, 'GM_*_scores.txt')):
		os.remove(f)

	# Delete temporary log files and create new logger
	for f in glob.glob(os.path.join(const.LOG_DIR, '*.tmp')):
		os.remove(f)
	LOG = logging.Logger()

	# Set parameters
	kernels  = ['linear', 'rbf']
	parallel = True
	nr_perm  = 1
	nr_iter  = 10

	# Get subject IDs and corresponding labels
	labels = load_roi(const.ROI_FILES_SUBCORTICAL[0], 'HC', 'SZ')['diagnosis']
	ids = util.get_bootstrapped_ids(labels, nr_perm)
	with open('ids.json', 'w') as f:
		json.dump(ids, f)

	for i in range(nr_perm):

		LOG.info('Running permutation {}'.format(i))
		start = timing.now()

		# Run classifier on each ROI
		threads = []
		for roi_name in const.ROI_NAMES_SUBCORTICAL:
			LOG.info('  {}'.format(roi_name))
			if not parallel:
				run_gp(roi_name, i, kernels, nr_iter)
			else:
				thread = multiprocessing.Process(
					target=run_gp, args=(roi_name, i, kernels, nr_iter))
				threads.append(thread)
				thread.start()
		if parallel:
			for thread in threads:
				thread.join()

		LOG.info('Time elapsed: {}'.format(timing.elapsed(start)))

	# Remove JSON file
	if os.path.isfile('ids.json'):
		os.remove('ids.json')

	# Close log
	LOG.append_file(__file__)
	LOG.close()


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	run()