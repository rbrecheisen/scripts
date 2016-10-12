#!/usr/bin/python

import os
import sys

#--------------------------------------------------------------------------------------------------
# Check and get input arguments
if len(sys.argv) != 6:

	print('')
	print('Usage: fslvbm_create_feature_tables.py <roidir> <roifile> <subjectsfile> <voxelinfodir> <outputdir>')
	print('')
	print('       <roidir>       : directory containing ROIs')
	print('       <roifile>      : file with ROI names')
	print('       <subjectsfile> : file with subject IDs and labels')
	print('       <voxelinfodir> : directory containing <subject>_<roi>_values.txt files')
	print('       <outputdir>    : directory where to save feature tables')
	print('')

	sys.exit(1)

roi_dir         = sys.argv[1]
roi_file        = sys.argv[2]
subject_id_file = sys.argv[3]
voxel_info_dir  = sys.argv[4]
output_dir      = sys.argv[5]

#--------------------------------------------------------------------------------------------------
# Load ROIs
rois = []
f = open(roi_dir + '/' + roi_file, 'r')
for roi in f.readlines():
	roi = roi[:-1]
	if roi.startswith('#') or roi == '':
		continue
	rois.append(roi)
f.close()

#--------------------------------------------------------------------------------------------------
# Load subjects and labels
subjects = []
subject_labels = []

f = open(subject_id_file, 'r')

for line in f.readlines():

	line = line[:-1]
	if line.startswith('#') or line == '':
		continue

	parts = line.split(',')

	if not len(parts) == 2:
		print('error: line does not match <subject>,<label> format')
		sys.exit(0)

	subject = parts[0]
	subject_label = parts[1]
	subjects.append(subject)
	subject_labels.append(subject_label)

f.close()

#--------------------------------------------------------------------------------------------------
# Process each ROI
for roi in rois:

	table  = []
	header = []

	for i in range(len(subjects)):

		subject = subjects[i]
		subject_label = subject_labels[i]
		values_file = voxel_info_dir + '/' + subject + '_' + roi + '_values.txt'

		if not os.path.isfile(values_file):
			print('error: file {} does not exist'.format(values_file))
			continue
			
		f = open(values_file, 'r')
		f.readline()
		f.readline()
		f.readline()

		values = []
		values.append(subject)
		values.extend(f.readline()[:-1].strip().split('  '))
		values.append(subject_label)
		
		table.append(values)

		if len(header) == 0:
			header.append('id')
			for i in range(len(values)):
				header.append('V' + str(i))
			header.append('diagnosis')
			print('processing: {} ({} features)'.format(roi, len(values)))

		f.close()

	table_file = output_dir + '/features_' + roi + '.txt'

	f = open(table_file, 'w')
	f.write(','.join(header) + '\n')
	for item in table:
		f.write(','.join(item) + '\n')
	f.close()

print('done')