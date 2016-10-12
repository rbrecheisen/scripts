__author__ = 'Ralph'

TIMESTAMP = '20150922_154700'
BASE_DIR = '/Users/Ralph/experiments/oslo/experiments/20150901_gp'
LOG_FILE = '{}/logs/{}.txt'.format(BASE_DIR, TIMESTAMP)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

sns.set_style('whitegrid')

with open(LOG_FILE, 'r') as f:

	mode = ''
	scores = []
	groups = []
	scores_hc_sz = []
	scores_hc_bd = []
	scores_sz_bd = []

	if not os.path.isdir('figs'):
		os.mkdir('figs')

	for line in f.readlines():

		if not line.startswith('2015'):
			continue

		line = line.strip()

		if 'HC - SZ' in line:
			mode = 'HC - SZ'
		elif 'HC - BD' in line:
			mode = 'HC - BD'
		elif 'SZ - BD' in line:
			mode = 'SZ - BD'

		if not '   score' in line:
			continue

		parts = line.strip().split(' ')
		score = float(parts[6])

		if mode == 'HC - SZ':
			scores_hc_sz.append(score)
		elif mode == 'HC - BD':
			scores_hc_bd.append(score)
		elif mode == 'SZ - BD':
			scores_sz_bd.append(score)

		scores.append(score)
		groups.append(mode)

	s1 = pd.Series(np.array(scores), dtype=np.float)
	s2 = pd.Series(np.array(groups))
	df = pd.DataFrame({'scores': s1, 'groups': s2})

	sns.barplot(x="groups", y="scores", data=df)
	plt.title('GP scores (RBF kernel)')
	plt.savefig(os.path.join(BASE_DIR, 'figs', TIMESTAMP + '_scores.png'), transparent=True)

	print('HC - SZ: {}'.format(np.mean(scores_hc_sz)))
	print('HC - BD: {}'.format(np.mean(scores_hc_bd)))
	print('SZ - BD: {}'.format(np.mean(scores_sz_bd)))