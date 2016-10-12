# ----------------------------------------------------------------------------------------------------------------------
# This script generates a statistical report from the output.
# ----------------------------------------------------------------------------------------------------------------------
__author__ = 'Ralph Brecheisen'

import os
import sys

sys.path.insert(1, os.path.abspath('../..'))

from scripts import util
from scripts import const
from scripts import logging
from scripts import evaluate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

from scipy.stats import ttest_ind
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon

RESULTS_FILE = '/Users/Ralph/experiments/oslo/experiments/20150818_perf_eval/results/20150930_104309_perf_info.json'
REPORT = logging.Logger(log_dir=const.REPORTS_DIR)


# ----------------------------------------------------------------------------------------------------------------------
def chunk(l, n):

	return [l[i:i+n] for i in range(0, len(l), n)]


# ----------------------------------------------------------------------------------------------------------------------
def create_df(accuracies):

	scores = []
	classifiers = []
	for key in accuracies.keys():
		for accuracy in accuracies[key]:
			scores.append(accuracy)
			classifiers.append(str(key))

	df = pd.DataFrame({
		'scores': pd.Series(np.array(scores), dtype=float),
		'classifiers': pd.Series(np.array(classifiers))
	})

	return df

# ----------------------------------------------------------------------------------------------------------------------
def create_plot(accuracies):

	mean_scores = []
	for key in accuracies.keys():
		mean_scores.append(np.mean(accuracies[key]))
	stdv_scores = []
	for key in accuracies.keys():
		stdv_scores.append(np.std(accuracies[key]))
	x = np.arange(len(accuracies))
	width = 0.35
	fig, ax = plt.subplots()
	rectangles = ax.bar(x, mean_scores, width, color='r', yerr=stdv_scores)
	ax.set_ylabel('Scores')
	ax.set_title('Accuracy scores')
	ax.set_xticks(x + width)
	ax.set_xticklabels(tuple(accuracies.keys()))
	for i in range(len(rectangles)):
		rect = rectangles[i]
		height = rect.get_height()
		accuracy = mean_scores[i]
		ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, '{0:.3f}'.format(accuracy), ha='center', va='bottom')
	plt.savefig('figs/scores1.png', transparent=True)

# ----------------------------------------------------------------------------------------------------------------------
def run(file_name):

	results = util.from_file(file_name)

	# Calculate accuracies for each classifier based on the true and
	# predicted labels
	accuracies = {}
	for classifier in results.keys():
		accuracies[classifier] = []
		for info in results[classifier]:
			y_true = info['true']
			y_pred = info['pred']
			accuracies[classifier].append(evaluate.get_accuracy_score(y_true, y_pred))

	# Create plot of scores
	if not os.path.isdir('figs'):
		os.mkdir('figs')
	df = create_df(accuracies)
	sns.barplot(x='classifiers', y='scores', data=df)
	plt.savefig('figs/scores.png', transparent=True)
	create_plot(accuracies)

	# Print mean accuracies and standard deviations
	REPORT.info()
	REPORT.info('Accuracies:')
	for classifier in accuracies.keys():
		REPORT.info('{}: {} +/- {}'.format(classifier, np.mean(accuracies[classifier]), np.std(accuracies[classifier])))
	REPORT.info()
	
	# Run matched-samples, equal-variance t-test
	REPORT.info('Pairwise t-test:')
	p_value_min = 1.0
	classifiers_min = []
	for classifier1 in accuracies.keys():
		for classifier2 in accuracies.keys():
			scores1 = accuracies[classifier1]
			scores2 = accuracies[classifier2]
			p_value = ttest_ind(scores1, scores2, equal_var=True)[1]
			if p_value < p_value_min:
				p_value_min = p_value
				classifiers_min = [classifier1, classifier2]
			REPORT.info('{} <-> {}: {}'.format(classifier1, classifier2, p_value))
	REPORT.info('Min. p-value: {} between {} and {}'.format(p_value_min, classifiers_min[0], classifiers_min[1]))
	REPORT.info()

	# Run pairwise Wilcoxon ranked-sign test
	REPORT.info('Pairwise Wilcoxon ranked-sign test:')
	p_value_min = 1.0
	classifiers_min = []
	p_values = []
	classifiers = []
	for classifier1 in accuracies.keys():
		for classifier2 in accuracies.keys():
			if classifier1 == classifier2:
				continue
			scores1 = accuracies[classifier1]
			scores2 = accuracies[classifier2]
			p_value = wilcoxon(scores1, scores2, zero_method='wilcox')[1]
			if p_value < p_value_min:
				p_value_min = p_value
				classifiers_min = [classifier1, classifier2]
			if p_value < 0.05:
				p_values.append(p_value)
				classifiers.append([classifier1, classifier2])
			REPORT.info('{} <-> {}: wilcoxon p-value: {}'.format(classifier1, classifier2, p_value))
	REPORT.info('Min. p-value: {} between {} and {}'.format(p_value_min, classifiers_min[0], classifiers_min[1]))
	REPORT.info('Below 0.05:')
	for i in range(len(p_values)):
		REPORT.info('  p-value: {} between {} and {}'.format(p_values[i], classifiers[i][0], classifiers[i][1]))
	REPORT.info()

	# Run Friedman test
	REPORT.info('Friedman\'s Chi-Square test:')
	data = []
	for classifier in accuracies.keys():
		row = []
		for item in chunk(accuracies[classifier], 10):
			row.append(np.mean(item))
		data.append(row)
	data = pd.DataFrame(
		np.array(data).T,
		columns=accuracies.keys(),
		index=['D' + str(i) for i in range(10)])

	p_value = friedmanchisquare(
		data['svm-rbf'],
		data['svm-poly'],
		data['svm-sigmoid'],
		data['random-forest'],
		data['gp'])[1]
	REPORT.info()
	REPORT.info('p_value: {}'.format(p_value))

	# The p-value tells us IF there's a significant difference between
	# performance scores. The Q statistic allows us to determine which
	# classifiers are best.
	ranks = []
	for row in data.index:
		ranked = np.array(data.loc[row].tolist()).argsort()[::-1]
		ranks.append(ranked)
	ranks = pd.DataFrame(ranks, columns=data.columns, index=data.index)

	n = ranks.shape[0]
	k = ranks.shape[1]

	mean_ranks = []
	for column in ranks.columns:
		mean_ranks.append(np.mean(ranks[column]))
	mean_ranks = pd.DataFrame([mean_ranks], columns=ranks.columns, index=['mean_ranks'])
	print(mean_ranks)

	# Define q threshold at alpha = 0.05 and df = (n-1)*(k-1)
	qq_threshold = 3.85 / np.sqrt(2)
	REPORT.info('Q-threshold: {}'.format(qq_threshold))
	tuples = []
	for column1 in ranks.columns:
		for column2 in ranks.columns:
			if column1 == column2:
				continue
			if (column1, column2) not in tuples:
				tuples.append((column1, column2))
			if (column2, column1) in tuples:
				continue
			r1 = mean_ranks[column1][0]
			r2 = mean_ranks[column2][0]
			qq = (r1 - r2) / np.sqrt((k * (k + 1.0)) / (6.0 * n))
			REPORT.info('{} <-> {}: q = {}'.format(column1, column2, qq))
			if qq > qq_threshold:
				REPORT.info('  q-statistic significant: {} > {}'.format(qq, qq_threshold))

	# Close the report file
	REPORT.append_file(__file__)
	REPORT.close()


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	run(RESULTS_FILE)