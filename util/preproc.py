import numpy as np
import pandas as pd

__author__ = 'Ralph Brecheisen'


# ----------------------------------------------------------------------------------------------------------------------
def add_feature_columns(features, columns, file_name):
    extra_data = pd.read_csv(file_name, index_col='id')
    extra_data = extra_data.loc[features.index]
    for column in columns:
        features[column] = extra_data[column]

    return features


# ----------------------------------------------------------------------------------------------------------------------
def normalize_across_subjects(features, exclude, verbose=False):
    # Select numeric columns, subtract mean and divide by standard deviation
    features_num = features.select_dtypes(include=[np.dtype(float), np.dtype(int)])
    for column in features_num.columns:
        if column in exclude:
            continue
        if verbose:
            print('  Normalizing feature {}...'.format(column))
        features[column] = (features[column] - np.mean(features[column])) / np.std(features[column])

    return features


# ----------------------------------------------------------------------------------------------------------------------
def remove_features(features, columns, verbose=False):
    for column in columns:
        if verbose:
            print('  Removing feature {}...'.format(column))
        features.drop(column, axis=1, inplace=True)

    return features


# ----------------------------------------------------------------------------------------------------------------------
def remove_constant_features(features, verbose=False):
    # Remove features that have only constant values
    for column in features.columns:
        if features[column].nunique() == 1:
            if verbose:
                print('  Removing constant feature {}...'.format(column))
            features.drop(column, axis=1, inplace=True)

    return features


# ----------------------------------------------------------------------------------------------------------------------
def remove_correlated_features(features, threshold=0.95, verbose=False):
    # Calculate correlation matrix for all features
    merge_dict = {}
    correlations = features.corr()

    # For feature, build list of other features that are highly
    # correlated with it. These lists are later used to remove certain
    # features from the original data frame.
    for i in range(len(correlations.index)):
        for j in range(i, len(correlations.columns)):
            if i == j:
                continue
            coefficient = correlations.iloc[i, j]
            if coefficient > threshold:
                if not merge_dict.has_key(correlations.index[i]):
                    merge_dict[correlations.index[i]] = []
                merge_dict[correlations.index[i]].append(correlations.columns[j])

    for key in merge_dict.keys():
        for column in merge_dict[key]:
            # Check if column has not already been dropped. Also, check if
            # column is not one of the other keys
            if column in features.columns and not column in merge_dict.keys():
                if verbose:
                    print('  Dropping feature {}...'.format(column))
                features.drop(column, axis=1, inplace=True)

    return features


# ----------------------------------------------------------------------------------------------------------------------
def match_ages(features, label1, label2, age_diff, nr_labels=3):
    diagnoses = ['HC', 'SZ', 'BD']
    diagnoses.remove(label1)
    diagnoses.remove(label2)
    label3 = diagnoses[0]

    # Select only the true schizophrenia patients, not the milder forms.
    features1 = features[features['diagnosis'] == label1]
    features2 = features[features['diagnosis'] == label2]
    features3 = features[features['diagnosis'] == label3]

    count1 = len(features1)
    count2 = len(features2)
    if count1 > count2:
        features_tmp = features1
        features1 = features2
        features2 = features_tmp

    subjects1 = list(features1.index)
    subjects2 = list(features2.index)
    subjects3 = list(features3.index)
    matched1 = []
    matched2 = []
    matched3 = []

    k = 0

    for i in range(len(features1)):

        # Get age of SZ patient
        subject1 = subjects1[i]
        age1 = features1['age'].loc[subject1]

        # Search through remaining list of HC subjects for a subject
        # with an age less than 1 year apart from the SZ patient
        for j in range(len(subjects2)):

            subject2 = subjects2[j]
            age2 = features2['age'].loc[subject2]

            # If absolute difference in age is less or equal to age_diff, add both
            # subjects to the list
            if np.absolute(age1 - age2) <= age_diff:

                matched1.append(subject1)
                matched2.append(subject2)

                # Also add a subject corresponding to the third if there is one
                if k < len(subjects3):
                    subject3 = subjects3[k]
                    matched3.append(subject3)
                    k += 1

                # Remove the HC subject from further consideration since it was
                # already matches with the current patient
                subjects2.remove(subject2)
                break

    features1 = features1.loc[matched1]
    features2 = features2.loc[matched2]
    features3 = features3.loc[matched3]

    if nr_labels == 3:
        return pd.concat([features2, features1, features3])
    else:
        return pd.concat([features2, features1])


# ----------------------------------------------------------------------------------------------------------------------
def residualize(features, confounds, target='diagnosis', verbose=False):
    # Get numeric features. This should exclude the target feature but we
    # will explicitly exclude it anyway.
    features_num = features.select_dtypes(include=[np.dtype(float), np.dtype(int)])

    for confound in confounds:

        # If confound is of type 'object' this means it's categorical. We need to
        # convert it to discrete numbers so we can perform regression on them
        if features[confound].dtype is np.dtype(object):

            # Create explicit copy of the values because we'll be changing categorical
            # values to numeric values and we don't want to change the input features
            confound_values = features[confound].copy(deep=True)

            # Check that we have binary labels
            if confound_values.nunique() is not 2:
                raise RuntimeError('Only binary labels supported')

            # Replace each label by its numerical equivalent. The first label
            # encountered will be set to zero (0), the second to one (1).
            i = 0
            for label in confound_values.unique():
                confound_values.replace(label, i, inplace=True)
                i += 1
        else:
            confound_values = features[confound]

        # Convert confound values to Numpy 2-dimensional array
        X = np.array([list(confound_values)])
        X = np.transpose(X)

        # Calculate regression residuals for each numeric feature (except the
        # confounds we're whose effect we're trying to get rid of and the target label)
        for column in features_num.columns:
            if column in confounds or column is target:
                continue
            if verbose:
                print('  Subtracting confound {} from feature {}...'.format(confound, column))
            y = features[column]
            residuals = y - LinearRegression().fit(X, y).predict(X)
            features[column] = residuals

    return features


# ----------------------------------------------------------------------------------------------------------------------
def categorical_to_numeric(features, column):
    """
    Converts given column from categorical (string) values to numbers.
    :param features: Feature set.
    :param column: Column.
    :return: Features.
    """
    labels = np.unique(features[column])
    label_nrs = range(1, len(labels) + 1)
    features[column].replace(labels, label_nrs, inplace=True)

    return features


# ----------------------------------------------------------------------------------------------------------------------
def dummy_encode_categorical_features(features, verbose=False):
    """
    For each categorical feature, this function expands the feature with dummy
    variables and assigns a 1-out-of-K value.
    :param features: Features set.
    :param verbose: Verbosity.
    :return:
    """
    pass


# ----------------------------------------------------------------------------------------------------------------------
def reduce_features(features, var_explained=0.9, n_components=0, verbose=False):
    """
    Performs feature reduction using PCA. Automatically selects nr. components
    for explaining min_var_explained variance.
    :param features: Features.
    :param var_explained: Minimal variance explained.
    :param n_components: Nr. of components.
    :param exclude_columns: Columns to exclude.
    :param verbose: Verbosity.
    :return: Reduced feature set.
    """
    if n_components == 0:
        # Run full PCA to estimate nr. components for explaining given
        # percentage of variance.
        estimator = RandomizedPCA()
        estimator.fit_transform(features)
        variance = 0.0
        for i in range(len(estimator.explained_variance_ratio_)):
            variance += estimator.explained_variance_ratio_[i]
            if variance > var_explained:
                n_components = i + 1
                if verbose:
                    print('{} % of variance explained using {} components'.format(var_explained, n_components))
                break
    # Re-run PCA with only estimated nr. components
    estimator = RandomizedPCA(n_components=n_components)
    features = estimator.fit_transform(features)
    return features


# ----------------------------------------------------------------------------------------------------------------------
def remove_outliers(features, max_fraction=0.1, min_fraction=0.25, verbose=False):
    """
    Remove outliers from feature set. Since this is an unsupervised approach we iterate
    over many nu/gamma settings for the one-class SVM. For each setting, a certain fraction
    of the subjects will be classified as outliers. For some settings, this fraction will
    be very large, e.g., 90% which is not realistic. For this reason, you can set a maximum
    fraction, e.g., 10%. Only those parameter combinations that result in 10% or less outliers
    are considered for further analysis. Within those combinations we simply count how often
    a given subject is classified as an outlier. We then use a minimum fraction to determine
    when a subject is truly an outlier.
    :param features:
    :param max_fraction: Upper bound on number of outliers allowed
    :param min_fraction: Lower bound on number of times a subject is classified as outlier
    :param verbose: Verbosity.
    :return: Filtered feature set
    """
    X, y = util.get_xy(
        features,
        target_column='diagnosis',
        exclude_columns=['age', 'gender', 'diagnosis'])

    subjects = {}
    nr_ok_fractions = 0

    for nu in np.linspace(0.01, 1.0, num=20):

        for gamma in [2 ** x for x in range(-15, 4, 2)]:

            # Train classifier
            classifier = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
            classifier.fit(X)
            y_pred = classifier.predict(X)

            # Calculate fraction of outliers
            count = 0.0
            for i in range(len(y_pred)):
                if y_pred[i] == -1:
                    count += 1.0
            fraction = count / len(y_pred)

            # If fraction is less than threshold run through list again to find
            # which subjects are considered outliers. Each outlying subject is
            # added to the table and its value incremented by one
            if fraction < max_fraction:
                nr_ok_fractions += 1
                for i in range(len(y_pred)):
                    if y_pred[i] == -1:
                        subject = features.index[i]
                        if subject not in subjects.keys():
                            subjects[subject] = 0
                        subjects[subject] += 1

    # Print number of times each subject is identified as outlier
    outliers = []
    for subject in subjects.keys():
        fraction = subjects[subject] / float(nr_ok_fractions)
        if fraction >= min_fraction:
            outliers.append(subject)

    # Remove outlying subjects

    if verbose:
        print('Removing {} outliers...'.format(len(outliers)))
    features.drop(outliers, axis=0, inplace=True)

    return features
