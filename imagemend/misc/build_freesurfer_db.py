import os
import pandas
import sqlite3

ROOT_DIR = '/Volumes/imagemend.zi-mannheim.de'
META_FILES = {
	'CIMH':  os.path.join(ROOT_DIR, 'CIMH/MKT_IMAGEMEND_META.xlsx'),
	'UNIBA': os.path.join(ROOT_DIR, 'UNIBA/NEW_UNIBA_ModifiedMetaData_29092015.xlsx'),
	'UNICH': os.path.join(ROOT_DIR, 'UNIBA/NEW_UNIBA_ModifiedMetaData_29092015.xlsx'),
	'UiO':   os.path.join(ROOT_DIR, 'UiO/NEW_15042014_UiO_ModifiedMetaData_08052014.xlsx'),
}
DATA_FILES = {
	'CIMH':  os.path.join(ROOT_DIR, 'UiO/CIMH_FreeSurfer/stats/CIMH_allROIFeatures_N67.csv'),
	'UNIBA': os.path.join(ROOT_DIR, 'UiO/UNIBA_FreeSurfer/stats_N412/UNIBA_allROIFeatures_N412.csv'),
	'UNICH': os.path.join(ROOT_DIR, 'UiO/UNICH_FreeSurfer/stats_N111/UNICH_allROIFeatures_N111.csv'),
	'UiO':   os.path.join(ROOT_DIR, 'UiO/UiO_FreeSurfer/stats/TOP_allROIFeatures_N664.csv'),
}
META_COLS = {
	'Age': 'Age [years]',
	'Gender': 'Gender [m/f]',
	'Diagnosis': 'Diagnosis',
}
IGNORE = '../data/uio/smri/ignore.csv'

# ----------------------------------------------------------------------------------------------------------------------
def load_cols_to_ignore():
	ignore = []
	f = open(IGNORE, 'r')
	f.readline()
	for col in f.readlines():
		ignore.append(col.strip())
	f.close()
	return ignore

# ----------------------------------------------------------------------------------------------------------------------
def load_meta_files():
	meta_data = {}
	for center in META_FILES.keys():
		data = pandas.read_excel(META_FILES[center], index_col=1, skiprows=[0,1,2])
		for sid in data.index:
			if sid not in meta_data.keys():
				meta_data[sid] = {}
				meta_data[sid]['Center'] = center
			for col in META_COLS.keys():
				value = data.loc[sid, META_COLS[col]]
				if col == 'Gender':
					value = str(value)
					if value.lower().startswith('m'):
						value = 'M'
					else:
						value = 'F'
				meta_data[sid][col] = value
		print('Loaded meta data for center {}'.format(center))
	return meta_data

# ----------------------------------------------------------------------------------------------------------------------
def load_data_files():
	subj_data = None
	ids = []
	for center in DATA_FILES.keys():
		data = pandas.read_csv(DATA_FILES[center])
		duplicates = []
		for idx in data.index:
			sid = data.loc[idx, 'MRid']
			sid = sid.split('_')[0] + '_' + sid.split('_')[1]
			data.set_value(idx, 'MRid', sid)
			if sid in ids:
				print('Found duplicate ID {}'.format(sid))
				duplicates.append(sid)
			ids.append(sid)
		data.set_index('MRid', drop=True, inplace=True)
		data.drop(duplicates, inplace=True)
		data['Center'] = center
		if subj_data is None:
			subj_data = data
		else:
			subj_data = pandas.concat([subj_data, data])
		print('Loaded FreeSurfer data for center {}'.format(center))
	subj_data.drop(load_cols_to_ignore(), axis=1, inplace=True)
	return subj_data

# ----------------------------------------------------------------------------------------------------------------------
def merge_meta_and_subj_data(meta_data, subj_data):
	for col in META_COLS.keys():
		subj_data[col] = None
	sids_to_remove = []
	for sid in subj_data.index:
		if sid not in meta_data.keys():
			print('WARNING: Subject {} of {} not in meta data'.format(sid, subj_data.loc[sid, 'Center']))
			sids_to_remove.append(sid)
			continue
		for col in META_COLS.keys():
			subj_data.set_value(sid, col, meta_data[sid][col])
	if len(sids_to_remove) > 0:
		subj_data.drop(sids_to_remove, inplace=True)
	return subj_data

# ----------------------------------------------------------------------------------------------------------------------
def load_subj_data_from_csv(file_name):
	subj_data = pandas.read_csv(file_name, index_col='id')
	return subj_data

# ----------------------------------------------------------------------------------------------------------------------
def write_subj_data_to_csv(subj_data, file_name):
	subj_data.to_csv(file_name, index=True, index_label='id')

# ----------------------------------------------------------------------------------------------------------------------
def write_subj_data_to_sql(subj_data, file_name):
	if os.path.isfile(file_name):
		os.remove(file_name)
	conn = sqlite3.connect(file_name)
	cursor = conn.cursor()
	sql  = 'CREATE TABLE FreeSurfer ('
	sql += 'ID text, '
	for col in subj_data.columns[:-4]:
		col = col.replace('.', '_')
		col = col.replace('-', '_')
		sql += col + ' real, '
	sql += 'Center text, '
	sql += 'Age real, '
	sql += 'Gender text, '
	sql += 'Diagnosis text)'
	print(sql)
	cursor.execute(sql)
	row = 1
	nr_rows = len(subj_data.index)
	for sid in subj_data.index:
		sql  = 'INSERT INTO FreeSurfer VALUES('
		sql += '\'' + sid + '\', '
		for col in subj_data.columns[:-4]:
			sql += str(subj_data.loc[sid, col]) + ', '
		sql += '\'' + subj_data.loc[sid, 'Center'] + '\', '
		sql += str(subj_data.loc[sid, 'Age']) + ', '
		sql += '\'' + subj_data.loc[sid, 'Gender'] + '\', '
		sql += '\'' + subj_data.loc[sid, 'Diagnosis'] + '\')'
		cursor.execute(sql)
		print('Added {} / {} rows'.format(row, nr_rows))
		row += 1
	conn.commit()
	conn.close()

# ----------------------------------------------------------------------------------------------------------------------
def run():
	meta_data = load_meta_files()
	subj_data = load_data_files()
	subj_data = merge_meta_and_subj_data(meta_data, subj_data)
	write_subj_data_to_csv(subj_data, 'freesurfer.csv')
	write_subj_data_to_sql(subj_data, 'freesurfer.db')

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	run()
