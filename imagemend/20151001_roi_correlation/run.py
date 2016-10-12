# ----------------------------------------------------------------------------------------------------------------------
# This script calculates pairwise correlations between ROIs
# ----------------------------------------------------------------------------------------------------------------------
__author__ = 'Ralph Brecheisen'

import os
import sys
import glob
import json
import multiprocessing

sys.path.insert(1, os.path.abspath('../..'))

from scripts import util
from scripts import const
from scripts import logging
from scripts import prepare

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------------------------------------------------
def load_roi(file_name):

	roi_data = util.load_features(file_name)
	meta_file = os.path.join(const.DATA_DIR, 'meta.txt')
	roi_data = prepare.add_feature_columns(roi_data, ['age', 'diagnosis'], meta_file)
	roi_data = prepare.match_ages(roi_data, 'HC', 'SZ', 2, nr_labels=2)

	return roi_data


# ----------------------------------------------------------------------------------------------------------------------
def calc_corr(roi_name1, roi_name2, coeffs):

	# Load ROI data and select only numeric columns
	roi_data1 = load_roi(const.to_file(roi_name1))
	roi_data1 = roi_data1.select_dtypes(include=[np.float])
	roi_data2 = load_roi(const.to_file(roi_name2))
	roi_data2 = roi_data2.select_dtypes(include=[np.float])

	# Calculate correlations between each feature vector pair
	coeff_tab = []
	for column1 in roi_data1.columns:
		coeff_row = []
		for column2 in roi_data2.columns:
			coeff = roi_data1[column1].corr(roi_data2[column2])
			coeff_row.append(coeff)
		coeff_tab.append(coeff_row)

	# Calculate mean correlation
	mean_corr = []
	for row in coeff_tab:
		for coeff in row:
			mean_corr.append(coeff)
	mean_corr = np.mean(mean_corr)

	key = roi_name1 + '_' + roi_name2
	coeffs[key] = mean_corr
	print('{}: {}'.format(key, coeffs[key]))


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

	# Define some parameters
	done = []
	threads = []
	parallel = True
	manager = multiprocessing.Manager()
	coeffs = manager.dict()

	# Run calculation
	for roi_name1 in const.ROI_NAMES_SUBCORTICAL:
		for roi_name2 in const.ROI_NAMES_SUBCORTICAL:
			if roi_name1 == roi_name2:
				continue
			if roi_name2 in done:
				continue
			LOG.info('{} <-> {}'.format(roi_name1, roi_name2))
			if parallel:
				thread = multiprocessing.Process(
					target=calc_corr, args=(roi_name1, roi_name2, coeffs))
				threads.append(thread)
				thread.start()
			else:
				calc_corr(roi_name1, roi_name2, coeffs)
		done.append(roi_name1)
	if parallel:
		for thread in threads:
			thread.join()

	# Print coefficients
	for key in coeffs.keys():
		print('{}: {}'.format(key, coeffs[key]))


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	run()