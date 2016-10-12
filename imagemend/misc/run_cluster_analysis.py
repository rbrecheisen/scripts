import os
import numpy
import pandas
import sqlite3
import pylab

import tsne

# ----------------------------------------------------------------------------------------------------------------------
def get_columns(cursor):

	cursor.execute('SELECT sql FROM sqlite_master WHERE tbl_name = \'FreeSurfer\' AND type = \'table\'')
	data = cursor.fetchone()
	data = data[0].replace('CREATE TABLE FreeSurfer (', '')
	data = data[:-1]
	data = [x.strip() for x in data.split(',')]
	columns = []
	for item in data:
		columns.append(item.split(' ')[0])
	return columns

# ----------------------------------------------------------------------------------------------------------------------
def to_data_frame(data, columns):

	data_frame = pandas.DataFrame(data, columns=columns)
	data_frame.set_index('ID', drop=True, inplace=True)
	return data_frame

# ----------------------------------------------------------------------------------------------------------------------
def normalize(data, exclude):

	data_num = data.select_dtypes(include=[numpy.dtype(float)])
	for column in data_num.columns:
		if column in exclude:
			continue
		data[column] = (data[column] - numpy.mean(data[column])) / numpy.std(data[column])
	return data

# ----------------------------------------------------------------------------------------------------------------------
def get_xy(data, label, exclude):

	predictors = list(data.columns)
	for column in exclude:
		predictors.remove(column)
	if not label in exclude:
		predictors.remove(label)
	X = data[predictors]
	y = data[label]
	return X.as_matrix(), y.as_matrix()

# ----------------------------------------------------------------------------------------------------------------------
def get_labels(data, label):

	return data[label].as_matrix()

# ----------------------------------------------------------------------------------------------------------------------
def labels_to_colors(labels):

	values = numpy.unique(labels).tolist()
	colors = {
		'01_red': [1, 0, 0],
		'02_green': [0, 1, 0],
		'03_blue': [0, 0, 1],
		'04_yellow': [1, 1, 0],
		'05_magenta': [1, 0, 1],
		'06_cyan': [0, 1, 1],
		'07_black': [0, 0, 0],
	}
	colors_idx = sorted(colors)
	value_colors = {}

	if len(values) > len(colors_idx):
		value_range = max(values) - min(values) + 1
		step = value_range / 4
		for value in values:
			if min(values) <= value < min(values) + step:
				value_colors[value] = colors[colors_idx[0]]
				continue
			if min(values) + step <= value < min(values) + 2*step:
				value_colors[value] = colors[colors_idx[1]]
				continue
			if min(values) + 2*step <= value < min(values) + 3*step:
				value_colors[value] = colors[colors_idx[2]]
				continue
			if min(values) + 3*step <= value < min(values) + 4*step:
				value_colors[value] = colors[colors_idx[3]]
				continue
	else:
		for i in range(len(values)):
			value_colors[values[i]] = colors[colors_idx[i]]

	label_colors = []
	for label in labels:
		label_colors.append(value_colors[label])
	return label_colors

# ----------------------------------------------------------------------------------------------------------------------
def save_csv(file_name, Y):

	f = open(file_name, 'w')
	for i in range(len(Y)):
		row = [str(x) for x in Y[i]]
		row = ','.join(row)
		f.write(row + '\n')
	f.close()

# ----------------------------------------------------------------------------------------------------------------------
def load_csv(file_name):

	Y = []
	f = open(file_name, 'r')
	for row in f.readlines():
		row = row.strip().split(',')
		row = [float(x) for x in row]
		Y.append(row)
	f.close()
	return numpy.array(Y)

# ----------------------------------------------------------------------------------------------------------------------
def run():

	# Setup connection to SQL database
	connection = sqlite3.connect('freesurfer.db')
	cursor = connection.cursor()

	# Extract pandas data frame from SQL query
	cursor.execute('SELECT * FROM FreeSurfer WHERE Center = \'UiO\' AND (Diagnosis = \'SZ\' OR Diagnosis = \'BD\' OR Diagnosis = \'HC\')')
	data = cursor.fetchall()
	data = to_data_frame(data, get_columns(cursor))
	data = normalize(data, exclude=['Age'])

	for target in ['Gender', 'Age']:

		# Run t-SNE procedure and visualize clusters
		X, _ = get_xy(data, target, exclude=['Diagnosis', 'Age', 'Center', 'Gender'])
		labels = get_labels(data, target)
		X = X.T

		if not os.path.isfile('Y.txt'):
			Y = tsne.tsne(X, 2, 50, 20.0)
			save_csv('Y.txt', Y)
		else:
			Y = load_csv('Y.txt')

		label_colors = labels_to_colors(labels)
		pylab.scatter(x=Y[:,0], y=Y[:,1], s=20, c=label_colors)
		pylab.title(target)
		pylab.show()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	run()
