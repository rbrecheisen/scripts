import os
import shutil
import numpy as np
import pandas as pd

SFTP_DIR = '/Volumes/imagemend-yJz2euONQq'
DATA_DIR = '/Users/Ralph/datasets/imagemend/multimodal'


def collect_subject_info():
    # Check if SFTP directory was mounted
    if not os.path.isdir(SFTP_DIR):
        print('Error: the SFTP directory is not mounted')
        return
    # Load subject ID and file paths
    subjects = {}
    for f in os.listdir(os.path.join(SFTP_DIR, 'UiO/rsfMRI')):
        subjects['_'.join(f.split('_')[0:4])] = {
            'diagnosis': '',
            'age': 0,
            'gender': '',
            'rs_path': os.path.join(SFTP_DIR, 'UiO/rsfMRI'),
            'rs_file': f
        }
    # Load subject meta data from Excel sheet
    meta_data = pd.read_excel(
        os.path.join(SFTP_DIR, 'UiO/NEW_15042014_UiO_ModifiedMetaData_08052014.xlsx'),
        header=3, index_col=1)
    # Search for each subject's diagnosis, age and gender
    for subject_id in subjects.keys():
        sid = '_'.join(subject_id.split('_')[0:2])
        mid = '_'.join(subject_id.split('_')[2:4])
        if sid in meta_data.index and meta_data.loc[sid][0] == mid:
            subjects[subject_id]['diagnosis'] = meta_data.loc[sid, 'Diagnosis']
            # Seems there are a few 'nan' values out there, so remove these
            # subjects from the dictionary
            age = meta_data.loc[sid, 'Age [years]']
            if np.isnan(age):
                del subjects[subject_id]
                continue
            subjects[subject_id]['age'] = age
            subjects[subject_id]['gender'] = meta_data.loc[sid, 'Gender [m/f]'][0].upper()
        else:
            print('RSFMRI: subject {}_{} not found in Excel sheet'.format(sid, mid))
    # Read structural MRI features
    stats_data = pd.read_csv(os.path.join(
        SFTP_DIR, 'UiO/UiO_TOP3T_FreeSurfer/stats/TOP3T_allROIFeatures_wmparcStatsAdded_N513.csv'), index_col=0)
    # For each resting-state subject check we also have structural features
    for subject_id in subjects.keys():
        if '{}_sMRI'.format(subject_id) not in stats_data.index:
            print('sMRI: subject {} not found in FreeSurfer stats file'.format(subject_id))
    return subjects


def print_summary(subjects):
    nr_males = 0
    nr_females = 0
    nr_bipolar = 0
    nr_schizophrenia = 0
    nr_healthy = 0
    nr_other = 0
    for k in subjects.keys():
        if subjects[k]['diagnosis'] == 'HC':
            nr_healthy += 1
        elif subjects[k]['diagnosis'] == 'BD':
            nr_bipolar += 1
        elif subjects[k]['diagnosis'] == 'SZ':
            nr_schizophrenia += 1
        else:
            nr_other += 1
        if subjects[k]['gender'] == 'M':
            nr_males += 1
        elif subjects[k]['gender'] == 'F':
            nr_females += 1
        else:
            pass
    print('males/females: {}/{}'.format(nr_males, nr_females))
    print('bipolar: {}'.format(nr_bipolar))
    print('schizophrenia: {}'.format(nr_schizophrenia))
    print('healthy: {}'.format(nr_healthy))
    print('other: {}'.format(nr_other))
    average_age = 0
    for k in subjects.keys():
        if subjects[k]['diagnosis'] == 'HC':
            average_age += subjects[k]['age']
    print('average age healthy: {}'.format(average_age / nr_healthy))
    average_age = 0
    for k in subjects.keys():
        if subjects[k]['diagnosis'] == 'BD':
            average_age += subjects[k]['age']
    print('average age bipolar: {}'.format(average_age / nr_bipolar))
    average_age = 0
    for k in subjects.keys():
        if subjects[k]['diagnosis'] == 'SZ':
            average_age += subjects[k]['age']
    print('average age schizophrenia: {}'.format(average_age / nr_schizophrenia))
    average_age = 0
    for k in subjects.keys():
        d = subjects[k]['diagnosis']
        if d != 'HC' and d != 'BD' and d != 'SZ':
            average_age += subjects[k]['age']
    print('average age other: {}'.format(average_age / nr_other))
    
    
def load_data(subjects):
    # Create local data directories
    rs_path = os.path.join(DATA_DIR, 'uio', 'rs')
    t1_path = os.path.join(DATA_DIR, 'uio', 't1')
    os.system('mkdir -p {}'.format(rs_path))
    os.system('mkdir -p {}'.format(t1_path))
    # Check if files have been copied already
    for k in subjects.keys():
        f = os.path.join(rs_path, subjects[k]['rs_file'])
        if not os.path.isfile(f):
            f = os.path.join(subjects[k]['rs_path'], subjects[k]['rs_file'])
            print('copying file {}'.format(f))
            shutil.copy(f, rs_path)
        subjects[k]['rs_path'] = rs_path
    return subjects


def run():
    subjects = collect_subject_info()
    print_summary(subjects)
    load_data(subjects)


if __name__ == '__main__':
    run()
