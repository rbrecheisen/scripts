import pandas as pd


config = {
    'CIMH': {
        'meta': '/Users/Ralph/data/imagemend/NEW_15042014_CIMH_ModifiedMetaData_15042014.xlsx',
        'features': '/Users/Ralph/data/imagemend/CIMH_allROIFeatures_N67.csv',
        'features_ext': None,
    },
    'UIO': {
        'meta': '/Users/Ralph/data/imagemend/NEW_15042014_UiO_ModifiedMetaData_08052014.xlsx',
        'features': '/Users/Ralph/data/imagemend/TOP_allROIFeatures_N664.csv',
        'features_ext': None,
    },
    'UNIBA': {
        'meta': '/Users/Ralph/data/imagemend/NEW_UNIBA_ModifiedMetaData_29092015.xlsx',
        'features': '/Users/Ralph/data/imagemend/UNIBA_allROIFeatures_N412.csv',
        'features_ext': None,
    },
    'UNICH': {
        'meta': '/Users/Ralph/data/imagemend/NEW_UNIBA_ModifiedMetaData_29092015.xlsx',
        'features': '/Users/Ralph/data/imagemend/UNICH_allROIFeatures_N111.csv',
        'features_ext': None,
    },
}


def to_gender(gender):
    gender = gender.upper()
    if gender not in ['M', 'F', 'MALE', 'FEMALE']:
        raise RuntimeError('Unknown gender encoding {}'.format(gender))
    if gender == 'M' or gender == 'F':
        return gender
    else:
        return gender[0]


def index_of(column, columns):
    # Returns index of first occurrence of 'column' in 'columns' or
    # -1 if column does not exist
    i = 0
    for c in columns:
        if c == column:
            return i
        i += 1
    return -1


def run():
    # For each center,
    # - Load CSV file
    # - Add some columns for age, gender, center and diagnosis
    # - Save updated CSV file
    for center in config.keys():

        # Load meta data and give it a new index based on measurement and subject ID
        meta_data_file_path = config[center]['meta']
        meta_data = pd.read_excel(meta_data_file_path, header=3)
        index = []
        for i in range(len(meta_data.index)):
            mid = meta_data.iloc[i][meta_data.columns[0]]
            sid = meta_data.iloc[i][meta_data.columns[1]]
            index.append('{}_{}_sMRI'.format(sid, mid))
        meta_data.set_index(pd.Series(index))

        # Load feature data
        features_file_path = config[center]['features']
        features = pd.read_csv(features_file_path)

        # Select rows in meta data corresponding to subject IDs in feature data
        meta_data = meta_data.loc[features.index]

        # Convert gender values to standardized format
        for i in range(len(meta_data.index)):
            gender = meta_data.iloc[i]['Gender [m/f]']
            meta_data.set_value(i, 'Gender [m/f]', to_gender(gender))

        # Add columns to original feature data
        features['Center'] = center
        features['Age'] = meta_data['Age [years]']
        features['Gender'] = meta_data['Gender [m/f]']
        features['Diagnosis'] = meta_data['Diagnosis']
        config[center]['features_ext'] = features

    # Concatenate feature data sets
    features = pd.concat([
        config['CIMH']['features_ext'],
        config['UIO']['features_ext'],
        config['UNIBA']['features_ext'],
        config['UNICH']['features_ext'],
    ])

    # Save concatenated feature data back to CSV file
    features.to_csv('features.txt')


if __name__ == '__main__':
    run()
