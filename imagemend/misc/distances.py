import numpy
import numpy.linalg
import scipy.spatial.distance


# ----------------------------------------------------------------------------------------------------------------------
class Distance(object):

	def calculate(self, features, idx1, idx2):
		pass


# ----------------------------------------------------------------------------------------------------------------------
class EuclideanDistance(Distance):

	def calculate(self, features, idx1, idx2):
		u = features.loc[idx1]
		v = features.loc[idx2]
		return scipy.spatial.distance.euclidean(u, v)


# ----------------------------------------------------------------------------------------------------------------------
class CosineDistance(Distance):

	def calculate(self, features, idx1, idx2):
		u = features.loc[idx1]
		v = features.loc[idx2]
		return scipy.spatial.distance.cosine(u, v)


# ----------------------------------------------------------------------------------------------------------------------
class ManhattanCityBlockDistance(Distance):

	def calculate(self, features, idx1, idx2):
		u = features.loc[idx1]
		v = features.loc[idx2]
		return scipy.spatial.distance.cityblock(u, v)


# ----------------------------------------------------------------------------------------------------------------------
class MahalanobisDistance(Distance):

	def __init__(self, cov_mat):
		self.__cov_mat = cov_mat

	def calculate(self, features, idx1, idx2):
		u = features.loc[idx1]
		v = features.loc[idx2]
		i = numpy.linalg.inv(self.__cov_mat)
		return scipy.spatial.distance.mahalanobis(u, v, i)


# ----------------------------------------------------------------------------------------------------------------------
class CorrelationDistance(Distance):

	def calculate(self, features, idx1, idx2):
		u = features.loc[idx1]
		v = features.loc[idx2]
		return scipy.spatial.distance.correlation(u, v)


# ----------------------------------------------------------------------------------------------------------------------
class ChebyshevDistance(Distance):

	def calculate(self, features, idx1, idx2):
		u = features.loc[idx1]
		v = features.loc[idx2]
		return scipy.spatial.distance.chebyshev(u, v)
