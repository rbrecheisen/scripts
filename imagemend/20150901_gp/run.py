# ----------------------------------------------------------------------------------------------------------------------
# This script runs a Gaussian Process classifier on the FreeSurfer features.
# ----------------------------------------------------------------------------------------------------------------------
__author__ = 'Ralph Brecheisen'

import os
import sys

sys.path.insert(1, os.path.abspath('../..'))

from scripts import util
from scripts import const
from scripts import prepare

import GPy
import numpy as np

from sklearn.cross_validation import StratifiedKFold

from matplotlib import pyplot as plt

LOG = util.Logger()


# ----------------------------------------------------------------------------------------------------------------------
def find_params(features):

	X, y = util.get_xy(
		features,
		target_column='diagnosis',
		exclude_columns=['age', 'gender', 'diagnosis'])
	Y = np.array([y]).T
	n = X.shape[1]

	if not os.path.isdir('fig'):
		os.mkdir('fig')

	for k in kernels:
		for length_scale in np.linspace(1.0, 10.0, num=10):
			kernel = kernels[k]
			kernel.lengthscale = length_scale
			classifier = GPy.models.GPClassification(X, Y, kernel)
			plt.matshow(classifier.kern.K(classifier.X))
			plt.savefig('fig/fig_{}_{}.png'.format(k, str(length_scale)))
			plt.close()


# ----------------------------------------------------------------------------------------------------------------------
def build_kernel(name, input_dim):

	if name == 'linear':
		return GPy.kern.Linear(input_dim)
	if name == 'rbf':
		return GPy.kern.RBF(input_dim, lengthscale=5.)
	if name == 'matern32':
		return GPy.kern.Matern32(input_dim, lengthscale=4.)
	if name == 'exponential':
		return GPy.kern.Exponential(input_dim, lengthscale=2.)
	return None

# ----------------------------------------------------------------------------------------------------------------------
def run_gp(features, n_iters=10):

	X, y = util.get_xy(
		features,
		target_column='diagnosis',
		exclude_columns=['age', 'gender', 'diagnosis'])

	kernels = ['rbf']
	scores = {}

	for k in kernels:

		LOG.info('Running kernel {}'.format(k))
		scores[k] = []

		for i in range(n_iters):

			for train, test in StratifiedKFold(y, n_folds=10):

				X_train = X[train]
				Y_train = np.array([y[train]]).T
				kernel = build_kernel(k, X_train.shape[1])
				classifier = GPy.models.GPClassification(X_train, Y_train, kernel)
				for _ in range(5):
					classifier.optimize()

				X_test = X[test]
				Y_test = np.array([y[test]]).T
				probs = classifier.predict(X_test)[0]
				error_rate, _, _, _, _ = GPy.util.classification.conf_matrix(probs, Y_test, show=False)
				scores[k].append(1.0 - error_rate)
				LOG.info('  score ({}): {}'.format(k, scores[k][-1]))

	return scores


# ----------------------------------------------------------------------------------------------------------------------
def run():

	n_iters = 1

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
	# Remove gender feature
	features = prepare.remove_features(features, ['age', 'gender'])
	# Replace HC and SZ labels with -1 and 1
	features['diagnosis'].replace(['HC', 'SZ'], [1, 0], inplace=True)

	LOG.info('HC - SZ')
	scores = run_gp(features, n_iters)
	for k in scores.keys():
		LOG.info('overall score ({}): {}'.format(k, np.mean(scores[k])))

	# Load FreeSurfer features
	features = util.load_features(const.FREESURFER_FILE)
	# Perform age-matching between healthy and schizophrenia patients
	features = prepare.match_ages(features, 'HC', 'BD', 2, nr_labels=2)
	# Remove constant features
	features = prepare.remove_constant_features(features)
	# Normalize numerical features across subjects (excluding 'age')
	features = prepare.normalize_across_subjects(features, exclude=['age'])
	# Subtract age and gender confounds
	features = prepare.residualize(features, ['age', 'gender', 'EstimatedTotalIntraCranialVol'])
	# Replace HC and SZ labels with -1 and 1
	features['diagnosis'].replace(['HC', 'BD'], [1, 0], inplace=True)

	LOG.info('HC - BD')
	scores = run_gp(features, n_iters)
	for k in scores.keys():
		LOG.info('overall score ({}): {}'.format(k, np.mean(scores[k])))

	# Load FreeSurfer features
	features = util.load_features(const.FREESURFER_FILE)
	# Perform age-matching between healthy and schizophrenia patients
	features = prepare.match_ages(features, 'SZ', 'BD', 2, nr_labels=2)
	# Remove constant features
	features = prepare.remove_constant_features(features)
	# Normalize numerical features across subjects (excluding 'age')
	features = prepare.normalize_across_subjects(features, exclude=['age'])
	# Subtract age and gender confounds
	features = prepare.residualize(features, ['age', 'gender', 'EstimatedTotalIntraCranialVol'])
	# Replace HC and SZ labels with -1 and 1
	features['diagnosis'].replace(['SZ', 'BD'], [1, 0], inplace=True)

	LOG.info('SZ - BD')
	scores = run_gp(features, n_iters)
	for k in scores.keys():
		LOG.info('overall score ({}): {}'.format(k, np.mean(scores[k])))

	# Close log
	LOG.append_file(__file__)
	LOG.close()


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	run()