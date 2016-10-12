import os
import numpy
import pandas
import matplotlib.pyplot as plt

import util
import prepare
import distances

FREESURFER_FILE      = '../data/uio/smri/freesurfer/features.csv'
FREESURFER_FILE_PREP = 'freesurfer_prep.csv'


# ----------------------------------------------------------------------------------------------------------------------
class DBSCAN(object):

	def __init__(self, epsilon, min_pts):
		self.__epsilon = epsilon
		self.__min_pts = min_pts
		self.__clusters = []
		self.__noise = []
		self.__processed = []
		self.__nr_neighbors = 0.0

	def epsilon(self):
		return self.__epsilon

	def set_epsilon(self, epsilon):
		self.__epsilon = epsilon

	def min_pts(self):
		return self.__min_pts

	def set_min_pts(self, min_pts):
		self.__min_pts = min_pts

	def clusters(self):
		return self.__clusters

	def noise(self):
		return self.__noise

	def run(self, features):
		# Check if the number of examples < minimum nr. of points.
		if features.shape[0] < self.min_pts():
			# Create single new cluster consisting of all rows in the
			# feature set and return it.
			pass
		# Create directory for saving PNGs
		if not os.path.isdir('fig'):
			os.mkdir('fig')
		# Reduce dimensionality of the dataset
		features = prepare.reduce_features(features, n_components=32)
		# # Get covariance matrix
		# cov_matrix = self.covariance_matrix(features, 'cov.csv')
		# # Get pairwise correlation matrix
		# corr_matrix = self.correlation_matrix(features, 'corr.csv')
		# values = corr_matrix.values.flatten()
		# plt.hist(values, 50, normed=1, facecolor='green')
		# plt.title('Correlation')
		# plt.show()
		# Get Euclidean distance matrix
		dist_matrix = self.distance_matrix(features, distances.EuclideanDistance(), 'dist_euclidean.csv')
		values = dist_matrix.values.flatten()
		plt.hist(values, 50, normed=1, facecolor='green')
		plt.title('Euclidean')
		plt.show()
		# # Get cosine angle distance matrix
		# dist_matrix = self.distance_matrix(features, distances.CosineDistance(), 'dist_cosine.csv')
		# values = dist_matrix.values.flatten()
		# plt.hist(values, 50, normed=1, facecolor='green')
		# plt.title('Cosine')
		# plt.show()
		# # Get Manhattan city block distance matrix
		# dist_matrix = self.distance_matrix(features, distances.ManhattanCityBlockDistance(), 'dist_city.csv')
		# values = dist_matrix.values.flatten()
		# plt.hist(values, 50, normed=1, facecolor='green')
		# plt.title('City Block')
		# plt.show()
		# # Get Mahalanobis distance matrix
		# dist_matrix = self.distance_matrix(features, distances.MahalanobisDistance(cov_matrix), 'dist_mahalanobis.csv')
		# values = dist_matrix.values.flatten()
		# plt.hist(values, 50, normed=1, facecolor='green')
		# plt.title('Mahalanobis')
		# plt.show()
		# # Get correlation distance matrix
		# dist_matrix = self.distance_matrix(features, distances.CorrelationDistance(), 'dist_corr.csv')
		# values = dist_matrix.values.flatten()
		# plt.hist(values, 50, normed=1, facecolor='green')
		# plt.title('Correlation')
		# plt.show()
		# # Get Chebyshev distance matrix
		# dist_matrix = self.distance_matrix(features, distances.ChebyshevDistance(), 'dist_chebyshev.csv')
		# values = dist_matrix.values.flatten()
		# plt.hist(values, 50, normed=1, facecolor='green')
		# plt.title('Chebyshev')
		# plt.show()

		# # Update epsilon to a fraction of the maximum distance
		# self.set_epsilon(0.1 * dist_max)
		# print('updated epsilon: {}'.format(self.epsilon()))
		# # Reset cluster and noise lists
		# self.__clusters = []
		# self.__noise = []
		# self.__processed = []
		# # Run through list of data points
		# for idx in features.index:
		# 	if idx not in self.__processed:
		# 		# Data point has not been processed yet so expand the cluster
		# 		self.expand_cluster(features, dist_matrix, idx)
		# 	if len(self.__processed) == features.shape[0]:
		# 		# We have processed all data points, so we can stop
		# 		break
		# print('average nr. neighbors: {}'.format(self.__nr_neighbors / features.shape[0]))
		return None

	def expand_cluster(self, features, dist_matrix, start_idx):
		# Get ids of all data points in the epsilon neighborhood of start_idx
		neighbors = []
		for idx in features.index:
			if idx == start_idx:
				continue
			distance = dist_matrix.loc[idx, start_idx]
			if distance < self.epsilon():
				neighbors.append(idx)
		# Update number of neighbors found so we can calculate average later
		self.__nr_neighbors += len(neighbors)
		# Check if data point is noise
		if len(neighbors) < self.min_pts():
			self.__noise.append(start_idx)
			self.__processed.append(start_idx)
			return
		# Start new cluster
		cluster = list()
		cluster.append(start_idx)
		self.__processed.append(start_idx)
		# Try to expand the cluster with this point's neighbors
		seeds = []
		for neighbor in neighbors:
			if neighbor not in self.__processed:
				self.__processed.append(neighbor)
				seeds.append(neighbor)
			else:
				if neighbor not in self.__noise:
					continue
				else:
					self.__noise.remove(neighbor)
			cluster.append(neighbor)

	def process_neighbors(self):
		pass

	def covariance_matrix(self, features, file_name):
		if not os.path.isfile(file_name):
			print('Calculating feature correlation matrix')
			cov_matrix = features.cov()
			cov_matrix.to_csv(file_name, index=True, index_label='id')
		else:
			cov_matrix = pandas.read_csv(file_name, index_col='id')
		return cov_matrix

	def correlation_matrix(self, features, file_name):
		if not os.path.isfile(file_name):
			print('Calculating feature correlation matrix')
			corr_matrix = features.corr()
			corr_matrix.to_csv(file_name, index=True, index_label='id')
		else:
			corr_matrix = pandas.read_csv(file_name, index_col='id')
		return corr_matrix

	def distance_matrix(self, features, metric, file_name):
		# Calculate distance matrix for all data points
		if not os.path.isfile(file_name):
			print('Calculating {} distance matrix'.format(metric))
			dist_matrix = numpy.zeros((features.shape[0], features.shape[0]))
			for i in range(features.shape[0]):
				for j in range(i):
					if i == j:
						continue
					# Get data point index labels and calculate distance between them
					idx1, idx2 = features.index[i], features.index[j]
					dist = metric.calculate(features, idx1, idx2)
					dist_matrix[i, j] = dist
					dist_matrix[j, i] = dist
			# Create pandas data frame from the distance matrix. Also save it to file
			# in case we need it later.
			dist_matrix = pandas.DataFrame(dist_matrix, index=features.index, columns=features.index)
			dist_matrix.to_csv(file_name, index=True, index_label='id')
		else:
			dist_matrix = pandas.read_csv(file_name, index_col='id')
		# Return result
		return dist_matrix

# ----------------------------------------------------------------------------------------------------------------------
def run():
	# If prepared file does not exist yet, create it
	if not os.path.isfile(FREESURFER_FILE_PREP):
		# Load FreeSurfer features
		features = util.load_features(FREESURFER_FILE)
		# Perform age-matching
		features = prepare.match_ages(features, 'HC', 'SZ', age_diff=2)
		# Remove constant features
		features = prepare.remove_constant_features(features)
		# Normalize numerical features across subjects (excluding 'age')
		features = prepare.normalize_across_subjects(features, exclude=['age'])
		# Residualize features for age, gender and total intracranial volume
		features = prepare.residualize(features, ['age', 'gender', 'EstimatedTotalIntraCranialVol'])
		# Remove highly correlated features
		features = prepare.remove_correlated_features(features)
		# Remove certain columns
		features = prepare.remove_features(features, ['diagnosis', 'age', 'gender'])
		# Write prepared freesurfer features back to file
		util.save_features(FREESURFER_FILE_PREP, features)
	else:
		# Load prepared features
		features = util.load_features(FREESURFER_FILE_PREP)

	# Run DBSCAN on features
	dbscan = DBSCAN(epsilon=1.0, min_pts=5)
	dbscan.run(features)

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	run()
