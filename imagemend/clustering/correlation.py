import util
import util.preproc
from sklearn import metrics
from sklearn.cluster import DBSCAN
import numpy as np

FILE_NAME = '/Users/Ralph/data/imagemend/features_ext_multi_center.csv'


def build_correlation_matrix(features):

    features = features.select_dtypes(include=[np.dtype(float), np.dtype(int)])
    matrix = np.zeros((len(features.index), len(features.index)))
    for i in range(len(features.index)):
        for j in range(i, len(features.index)):
            x = features.iloc[i]
            y = features.iloc[j]
            distance = 1.0 - (np.corrcoef(x, y)[0, 1] + 1.0) / 2.0
            matrix[i, j] = distance
    return matrix


def load():

    return util.load_features(FILE_NAME, index_col='MRid')


def preprocess(features):

    features = util.preproc.remove_constant_features(features)
    features = util.preproc.normalize_across_subjects(features, exclude=['Age'])
    features = util.preproc.select_rows(features, 'Diagnosis', ['HC', 'SZ', 'BD'])
    # features = util.preproc.select_rows(features, 'Center', ['UIO'])
    group1 = features[features['Diagnosis'] == 'BD']
    group2 = features[features['Diagnosis'] == 'SZ']
    features = util.preproc.match_ages_new(features, group1, group2, age_diff=2)
    group1 = features[features['Diagnosis'] == 'BD']
    group2 = features[features['Diagnosis'] == 'HC']
    features = util.preproc.match_ages_new(features, group1, group2, age_diff=2)
    features = util.preproc.regress_out(features, ['Age', 'Gender'], 'Diagnosis')
    features = util.preproc.remove_features(features, ['Age', 'Gender'])
    corr_matrix = build_correlation_matrix(features)
    return corr_matrix


def cluster(X):

    dbscan = DBSCAN(metric='precomputed')
    db = dbscan.fit(X)
    print(db.labels_)


def project():
    # Project high dimensional data points to 2D space for plotting
    pass


def run():

    features = load()
    corrmatr = preprocess(features)
    cluster(corrmatr)


if __name__ == '__main__':
    run()
