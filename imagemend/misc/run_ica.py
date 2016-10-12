__author__ = 'Ralph'

# ----------------------------------------------------------------------------------------------------------------------
# Imports

import os
import numpy as np
import pandas as pd
import sklearn.decomposition as decomposition
import nibabel

import util


# ----------------------------------------------------------------------------------------------------------------------
# Constants

GM_FILE       = 'swc1nu.nii.txt'
LOG_FILE      = '/data/raw_data/imagemend/uio/smri/log.txt'
SUBJECT_DIR   = '/data/raw_data/imagemend/uio/smri/raw'
SUBJECT_FILE  = '/data/raw_data/imagemend/uio/smri/subjects.csv'
COMPONENT_DIR = '/data/raw_data/imagemend/uio/smri/ica'

# GM_FILE       = 'swc1nu.nii.txt'
# LOG_FILE      = '../data/uio/smri/log.txt'
# SUBJECT_DIR   = '../data/uio/smri/normalized'
# SUBJECT_FILE  = '../data/uio/smri/subjects.csv'
# COMPONENT_DIR = '../data/uio/smri/ica'

LOG = open(LOG_FILE, 'w')


# ----------------------------------------------------------------------------------------------------------------------
def load_subject_ids():

	subject_ids = []
	f = open(SUBJECT_FILE, 'r')
	for subject_id in f.readlines():
		subject_ids.append(subject_id.strip())
	f.close()
	return subject_ids

# ----------------------------------------------------------------------------------------------------------------------
def to_dataframe(data, index, columns):

	columns = []
	for i in range(data.shape[1]):
		columns.append('X{}'.format(i))
	data_frame = pd.DataFrame(data, columns=columns, index=index)	
	return data_frame

# ----------------------------------------------------------------------------------------------------------------------
def load_data():

	i = 1
	xdim = 0
	ydim = 0
	zdim = 0
	data = []

	subject_ids = load_subject_ids()

	for subject_id in subject_ids:
		log('loading {} / {}'.format(i, len(subject_ids)))
		fi = open(os.path.join(SUBJECT_DIR, subject_id, GM_FILE), 'r')
		if i == 1:
			xpos = np.array(fi.readline().strip().split('  '), dtype=np.int)
			xdim = np.max(xpos) + 1
			ypos = np.array(fi.readline().strip().split('  '), dtype=np.int)
			ydim = np.max(ypos) + 1
			zpos = np.array(fi.readline().strip().split('  '), dtype=np.int)
			zdim = np.max(zpos) + 1
			log('  dim: {} x {} x {}'.format(xdim, ydim, zdim))
		else:
			fi.readline()
			fi.readline()
			fi.readline()
		data.append(fi.readline().strip().split('  '))
		fi.close()
		i += 1

	log('converting to float')
	data = np.array(data, dtype=np.float)

	log('centering data')
	data = center_data(data)

	log('converting to data frame')
	columns = []
	for i in range(data.shape[1]):
		columns.append('V{}'.format(i))
	data_frame = pd.DataFrame(data, columns=columns, index=subject_ids)

	# log('write to csv')
	# data_frame.to_csv(os.path.join(COMPONENT_DIR, 'voxels.txt'), index=True, index_label='id')

	return data_frame, xdim, ydim, zdim


# ----------------------------------------------------------------------------------------------------------------------
def center_data(X):

	n_rows, n_cols = X.shape
	X = X - X.mean(axis=0)
	X = X - X.mean(axis=1).reshape(n_rows, 1)
	return X


# ----------------------------------------------------------------------------------------------------------------------
def write_nifti(file_name, data, shape):

	data = data.reshape(shape)
	image = nibabel.Nifti1Image(data, np.eye(4))
	image.to_filename(file_name)


# ----------------------------------------------------------------------------------------------------------------------
def write_csv(file_path, data):

	np.savetxt(file_path, data, delimiter=',', fmt='%.8f')


# ----------------------------------------------------------------------------------------------------------------------
def log(message, close=False):

	print(message)
	LOG.write(message + '\n')
	if close:
		LOG.close()


# ----------------------------------------------------------------------------------------------------------------------
def run_ica():

	log('loading data')
	start = util.now()
	voxels, xdim, ydim, zdim = load_data()
	log('  elapsed: {}'.format(util.elapsed(start)))

	log('running independent component analysis')
	start = util.now()
	ica = decomposition.FastICA(n_components=64, max_iter=200)
	sources = ica.fit_transform(voxels)
	sources = to_dataframe(sources, load_subject_ids(), ['X{}'.format(i) for i in range(64)])
	log('  elapsed: {}'.format(util.elapsed(start)))

	log('calculating correlations between voxel and component time courses')
	start = util.now()
	correlations = []
	for voxel in voxels.columns[:32]:
		voxel = voxels[voxel]
		max_correlation = 0
		for source in sources.columns:
			source = sources[source]
			correlation = np.corrcoef(voxel, source)
			if correlation > max_correlation:
				max_correlation = correlation
		correlations.append(max_correlation)
	log('  elapsed: {}'.format(util.elapsed(start)))

	# log('writing source signals to csv file')
	# start = util.now()
	# write_csv(os.path.join(COMPONENT_DIR, 'sources.txt'), X_new)
	# log('  elapsed: {}'.format(util.elapsed(start)))
	
	# log('writing components to nifti file')
	# if not os.path.isdir(COMPONENT_DIR):
	# 	os.mkdir(COMPONENT_DIR)
	# start = util.now()
	# i = 1
	# for component in components:
	# 	log('  writing {} / {}'.format(i, len(components)))
	# 	write_nifti('c{}.nii.gz'.format(i), component, (xdim, ydim, zdim))
	# 	i += 1
	# log('  elapsed: {}'.format(util.elapsed(start)))
	
	# log('writing components to csv file')
	# start = util.now()
	# write_csv(os.path.join(COMPONENT_DIR, 'components.txt'), components)
	# log('  elapsed: {}'.format(util.elapsed(start)))
	# log('done', close=True)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	run_ica()
