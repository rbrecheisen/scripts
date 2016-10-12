# ----------------------------------------------------------------------------------------------------------------------
# This script contains general utility functions
# ----------------------------------------------------------------------------------------------------------------------
__author__ = 'Ralph Brecheisen'

import pandas as pd
# import const
import json
import string
import random
import time
import os
import glob
import GPy

from sklearn.cross_validation import StratifiedKFold


# ----------------------------------------------------------------------------------------------------------------------
def load_features(file_name):

	features = pd.read_csv(file_name, index_col='id')
	return features


# ----------------------------------------------------------------------------------------------------------------------
def load_features_for(file_name, labels):

	features = load_features(file_name)
	tmp = []
	for label in labels:
		tmp.append(features[features[const.TARGET] == label])
	return pd.concat(tmp)


# ----------------------------------------------------------------------------------------------------------------------
def save_features(file_name, features):

	features.to_csv(file_name, index=True, index_label='id')

# ----------------------------------------------------------------------------------------------------------------------
def get_bootstrapped_ids(labels, nr_perm, nr_iter=10):

	table = {}
	for i in range(nr_perm):
		table[i] = {}
		for j in range(nr_iter):
			table[i][j] = {}
			k = 0
			for train, test in StratifiedKFold(labels, n_folds=10, shuffle=True):
				table[i][j][k] = {}
				train_ids = np.random.choice(train, len(train), True)
				table[i][j][k]['train'] = list(train_ids)
				test_ids = np.random.choice(test, len(test), True)
				table[i][j][k]['test']  = list(test_ids)
				k += 1
	return table


# ----------------------------------------------------------------------------------------------------------------------
def get_xy(features, target_column=None, exclude_columns=list()):

	predictors = list(features.columns)
	for column in exclude_columns:
		if column in predictors:
			predictors.remove(column)
	if target_column:
		if not target_column in exclude_columns:
			if target_column in predictors:
				predictors.remove(target_column)

	X = features[predictors]
	X = X.as_matrix()

	if target_column:
		y = features[target_column]
		y = y.as_matrix()
	else:
		y = None

	return X, y


# ----------------------------------------------------------------------------------------------------------------------
def build_gp_kernel(name, input_dim):

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
def split_file_name(file_name):

	parts = file_name.split('.')

	return parts[0], parts[1]


# ----------------------------------------------------------------------------------------------------------------------
def to_file(file_name, dictionary, sort_keys=False):

	f = open(file_name, 'w')
	json.dump(dictionary, f, indent=4, sort_keys=sort_keys)
	f.close()


# ----------------------------------------------------------------------------------------------------------------------
def from_file(file_name):

	f = open(file_name, 'r')
	dictionary = json.load(f)
	f.close()

	return dictionary


# ----------------------------------------------------------------------------------------------------------------------
def random_string(nr_chars=8):

	return ''.join(random.choice(string.digits) for i in range(nr_chars))


# ----------------------------------------------------------------------------------------------------------------------
def remove_files(reg_exp):

	for f in glob.glob(reg_exp):
		os.remove(f)


# ----------------------------------------------------------------------------------------------------------------------
def now():

	return time.time()


# ----------------------------------------------------------------------------------------------------------------------
def elapsed(start):

	delta = now() - start
	nr_hours = int(delta / 3600)
	nr_minutes = int((delta - nr_hours * 3600) / 60)
	nr_seconds = int((delta - nr_hours * 3600 - nr_minutes * 60))
	nr_hours = '0' + str(nr_hours) if nr_hours < 10 else str(nr_hours)
	nr_minutes = '0' + str(nr_minutes) if nr_minutes < 10 else str(nr_minutes)
	nr_seconds = '0' + str(nr_seconds) if nr_seconds < 10 else str(nr_seconds)

	return '{}:{}:{}'.format(nr_hours, nr_minutes, nr_seconds)


# ----------------------------------------------------------------------------------------------------------------------
class Logger(object):

	def info(self, message):
		print(message)

	def append_file(self, file_name):
		pass

	def close(self):
		pass
