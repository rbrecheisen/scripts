__author__ = 'Ralph Brecheisen'

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(1, os.path.abspath('../..'))

from scripts import util
from scripts import const
from scripts import prepare
from scripts import plotting
from scripts import clustering

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score

from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import MDS

import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
def calculate_distance_matrix(features, metric, file_name):
	if not os.path.isfile(file_name):
		pass
	else:
		dist_matrix = pd.read_csv(file_name, index_col='id')

	return None

# ----------------------------------------------------------------------------------------------------------------------
def calculate_quality(labels_true, labels_pred):

	if not len(labels_true) == len(labels_pred):
		raise RuntimeError('Nr. of predicted and true labels must be equal')

	print('Nr. clusters: {}'.format(len(np.unique(labels_pred))))
	score = adjusted_rand_score(labels_true, labels_pred)
	print('Adjusted Rand Index:        {}\t(range: [-1, 1])'.format(score))
	score = adjusted_mutual_info_score(labels_true, labels_pred)
	print('Adjusted Mutual Info Score: {}\t(range: [ 0, 1])'.format(score))
	score = homogeneity_score(labels_true, labels_pred)
	print('Homogeneity:                {}\t(range: [ 0, 1])'.format(score))
	score = completeness_score(labels_true, labels_pred)
	print('Completeness:               {}\t(range: [ 0, 1])\n'.format(score))

# ----------------------------------------------------------------------------------------------------------------------
def run():

	# DATA PREPARATION

	# Load FreeSurfer features
	features = util.load_features(const.FREESURFER_FILE)
	# Print histograms
	plotting.print_histograms([
		features[features['diagnosis'] == 'HC']['age'],
		features[features['diagnosis'] == 'SZ']['age'],
		features[features['diagnosis'] == 'BD']['age']],['HC', 'SZ', 'BD'], 2)
	# Perform age-matching
	features = prepare.match_ages(features, 'HC', 'SZ', age_diff=2, nr_labels=2)
	# Remove constant features
	features = prepare.remove_constant_features(features)
	# Normalize numerical features across subjects (excluding 'age')
	features = prepare.normalize_across_subjects(features, exclude=['age'])
	# Residualize features for age, gender and total intracranial volume
	features = prepare.residualize(features, ['age', 'gender', 'EstimatedTotalIntraCranialVol'])
	# Remove highly correlated features
	features = prepare.remove_correlated_features(features)
	# Save prepared features to file
	util.save_features('freesurfer_prep.csv', features)

	# CLUSTERING

	n_clusters = 2

	# Run K-mean mini-batch algorithm for 3 clusters. In this first run, we would
	# like to see whether the clustering algorithm is able to reproduce the
	# diagnostic categories. First we do this without the 'diagnosis' feature.
	X, _ = util.get_xy(features, exclude_columns=['diagnosis', 'age', 'gender'])
	print('Mini-Batch K-Means:')
	estimator = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters)
	estimator.fit(X)
	calculate_quality(features['diagnosis'], estimator.labels_)

	# Now, let's include 'diagnosis' as a feature and see if the algorithm
	# uses it to improve the clustering. For this, we copy the feature set
	# and convert the diagnostic labels to numeric values because that's
	# what K-means requires.
	features_new = prepare.copy_features(features)
	features_new = prepare.categorical_to_numeric(features_new, 'diagnosis')
	X, _ = util.get_xy(features_new, exclude_columns=['age', 'gender'])

	# Run clustering again and calculate quality measures
	print('Mini-Batch K-Means (with diagnosis):')
	estimator = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters)
	estimator.fit(X)
	calculate_quality(features_new['diagnosis'], estimator.labels_)

	# Recreate design matrix X without diagnosis column
	X, _ = util.get_xy(features, exclude_columns=['diagnosis', 'age', 'gender'])

	# Next step, spectral clustering
	print('Spectral clustering:')
	estimator = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
	estimator.fit(X)
	calculate_quality(features['diagnosis'], estimator.labels_)

	print('Hierarchical clustering:')
	estimator = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
	estimator.fit(X)
	calculate_quality(features['diagnosis'], estimator.labels_)

	# Try to run DBSCAN on the data. Noisy samples should get a label -1.
	print('DBSCAN:')
	X, _ = util.get_xy(features, exclude_columns=['diagnosis', 'age', 'gender'])
	for eps in [5, 10, 20, 50, 100]:
		estimator = DBSCAN(eps=eps, min_samples=5)
		estimator.fit(X)
		print('Nr. clusters: {}'.format(len(np.unique(estimator.labels_))))

	# MANIFOLD LEARNING

	# Get feature data matrix and run 50-component PCA on it
	X, _ = util.get_xy(features, exclude_columns=['diagnosis', 'age', 'gender'])
	X = prepare.reduce_features(X, n_components=50)

	# Run TSNE
	embedding = TSNE(2, init='random')
	Y = embedding.fit_transform(X)
	plt.scatter(Y[:,0], Y[:,1])
	plt.title('{}'.format(embedding.__class__.__name__))
	plt.show()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	run()
