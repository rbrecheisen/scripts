# ----------------------------------------------------------------------------------------------------------------------
# This script contains constants and lists of ROI names
# ----------------------------------------------------------------------------------------------------------------------
__author__ = 'Ralph Brecheisen'

import os

# DIRECTORIES

ROOT_DIR = os.path.abspath('../..')

SCRIPT_DIR = os.path.join(ROOT_DIR, 'scripts')

DATA_DIR = os.path.join(ROOT_DIR, 'data/uio/smri')

LOG_DIR = 'logs'

RESULTS_DIR = 'results'

REPORTS_DIR = 'reports'

FREESURFER_DIR = os.path.join(DATA_DIR, 'freesurfer')

ICA_DIR = os.path.join(DATA_DIR, 'ica')

ROI_DIR = os.path.join(DATA_DIR, 'rois')

OTHER_DIR = os.path.join(DATA_DIR, 'other')

# FILES

FREESURFER_FILE = os.path.join(FREESURFER_DIR, 'features.csv')

FREESURFER_ALL_FILE = os.path.join(FREESURFER_DIR, 'features_all.csv')

VOLUMES_FILE = os.path.join(DATA_DIR, 'volumes.csv')

IGNORE_FILE = os.path.join(DATA_DIR, 'ignore.csv')

WDBC_FILE = os.path.join(OTHER_DIR, 'wdbc.txt')

# OTHER VARS

CONFOUNDS = [
	'age',
	'gender',
]

TARGET = 'diagnosis'

CLASSES = [
	'HC',
	'SZ',
	'BD',
]

VOLUMES = []
with open(VOLUMES_FILE, 'r') as f:
	for line in f.readlines():
		VOLUMES.append(line.strip())

IGNORE = []
with open(IGNORE_FILE, 'r') as f:
	for ignore in f.readlines():
		IGNORE.append(ignore.strip())

ROI_NAMES_ALL = [
	'GM_accumbens_L',
	'GM_accumbens_R',
	'GM_amygdala_L',
	'GM_amygdala_R',
	'GM_angular_gyrus',
	'GM_brainstem',
	'GM_caudate_L',
	'GM_caudate_R',
	'GM_central_opercular_cortex',
	'GM_cingulate_gyrus_anterior',
	'GM_cingulate_gyrus_posterior',
	'GM_cuneal_cortex',
	'GM_frontal_medial_cortex',
	'GM_frontal_operculum_cortex',
	'GM_frontal_orbital_cortex',
	'GM_frontal_pole',
	'GM_heschls_gyrus',
	'GM_hippocampus_L',
	'GM_hippocampus_R',
	'GM_inferior_frontal_gyrus_pars_opercularis',
	'GM_inferior_frontal_gyrus_pars_triangularis',
	'GM_inferior_temporal_gyrus_anterior',
	'GM_inferior_temporal_gyrus_posterior',
	'GM_inferior_temporal_gyrus_temporooccipital',
	'GM_insular_cortex',
	'GM_intracalcarine_cortex',
	'GM_lateral_occipital_cortex_inferior',
	'GM_lateral_occipital_cortex_superior',
	'GM_lingual_gyrus',
	'GM_middle_frontal_gyrus',
	'GM_middle_temporal_gyrus_anterior',
	'GM_middle_temporal_gyrus_posterior',
	'GM_middle_temporal_gyrus_temporooccipital',
	'GM_occipital_fusiform_gyrus',
	'GM_occipital_pole',
	'GM_pallidum_L',
	'GM_pallidum_R',
	'GM_paracingulate_gyrus',
	'GM_parahippocampal_gyrus_anterior',
	'GM_parahippocampal_gyrus_posterior',
	'GM_parietal_operculum_cortex',
	'GM_planum_polare',
	'GM_planum_temporale',
	'GM_postcentral_gyrus',
	'GM_precentral_gyrus',
	'GM_precuneus_cortex',
	'GM_putamen_L',
	'GM_putamen_R',
	'GM_subcallosal_cortex',
	'GM_superior_frontal_gyrus',
	'GM_superior_parietal_lobule',
	'GM_superior_temporal_gyrus_anterior',
	'GM_superior_temporal_gyrus_posterior',
	'GM_supplementary_motor_cortex',
	'GM_supracalcarine_cortex',
	'GM_supramarginal_gyrus_anterior',
	'GM_supramarginal_gyrus_posterior',
	'GM_temporal_fusiform_cortex_anterior',
	'GM_temporal_fusiform_cortex_posterior',
	'GM_temporal_occipital_fusiform_cortex',
	'GM_temporal_pole',
	'GM_thalamus_L',
	'GM_thalamus_R',
]

ROI_NAMES_CORTICAL = [
	'GM_angular_gyrus',
	'GM_central_opercular_cortex',
	'GM_cingulate_gyrus_anterior',
	'GM_cingulate_gyrus_posterior',
	'GM_cuneal_cortex',
	'GM_frontal_medial_cortex',
	'GM_frontal_operculum_cortex',
	'GM_frontal_orbital_cortex',
	'GM_frontal_pole',
	'GM_heschls_gyrus',
	'GM_inferior_frontal_gyrus_pars_opercularis',
	'GM_inferior_frontal_gyrus_pars_triangularis',
	'GM_inferior_temporal_gyrus_anterior',
	'GM_inferior_temporal_gyrus_posterior',
	'GM_inferior_temporal_gyrus_temporooccipital',
	'GM_insular_cortex',
	'GM_intracalcarine_cortex',
	'GM_lateral_occipital_cortex_inferior',
	'GM_lateral_occipital_cortex_superior',
	'GM_lingual_gyrus',
	'GM_middle_frontal_gyrus',
	'GM_middle_temporal_gyrus_anterior',
	'GM_middle_temporal_gyrus_posterior',
	'GM_middle_temporal_gyrus_temporooccipital',
	'GM_occipital_fusiform_gyrus',
	'GM_occipital_pole',
	'GM_paracingulate_gyrus',
	'GM_parahippocampal_gyrus_anterior',
	'GM_parahippocampal_gyrus_posterior',
	'GM_parietal_operculum_cortex',
	'GM_planum_polare',
	'GM_planum_temporale',
	'GM_postcentral_gyrus',
	'GM_precentral_gyrus',
	'GM_precuneus_cortex',
	'GM_subcallosal_cortex',
	'GM_superior_frontal_gyrus',
	'GM_superior_parietal_lobule',
	'GM_superior_temporal_gyrus_anterior',
	'GM_superior_temporal_gyrus_posterior',
	'GM_supplementary_motor_cortex',
	'GM_supracalcarine_cortex',
	'GM_supramarginal_gyrus_anterior',
	'GM_supramarginal_gyrus_posterior',
	'GM_temporal_fusiform_cortex_anterior',
	'GM_temporal_fusiform_cortex_posterior',
	'GM_temporal_occipital_fusiform_cortex',
	'GM_temporal_pole',
]

ROI_NAMES_SUBCORTICAL = [
	'GM_accumbens_L',
	'GM_accumbens_R',
	'GM_amygdala_L',
	'GM_amygdala_R',
	'GM_brainstem',
	'GM_caudate_L',
	'GM_caudate_R',
	'GM_hippocampus_L',
	'GM_hippocampus_R',
	'GM_pallidum_L',
	'GM_pallidum_R',
	'GM_putamen_L',
	'GM_putamen_R',
	'GM_thalamus_L',
	'GM_thalamus_R',
]

ROI_NAMES_MERGE_ALL = [
	'GM_merge_all',
]

ROI_NAMES_MERGE_CORTICAL = [
	'GM_merge_cortical',
]

ROI_NAMES_MERGE_SUBCORTICAL = [
	'GM_merge_subcortical',
]

ROI_FILES_ALL = []
for roi_name in ROI_NAMES_ALL:
	ROI_FILES_ALL.append(os.path.join(ROI_DIR, roi_name + '_s0.txt'))

ROI_FILES_CORTICAL = []
for roi_name in ROI_NAMES_CORTICAL:
	ROI_FILES_CORTICAL.append(os.path.join(ROI_DIR, roi_name + '_s0.txt'))

ROI_FILES_SUBCORTICAL = []
for roi_name in ROI_NAMES_SUBCORTICAL:
	ROI_FILES_SUBCORTICAL.append(os.path.join(ROI_DIR, roi_name + '_s0.txt'))

ROI_FILES_MERGE_ALL = []
for roi_name in ROI_NAMES_MERGE_ALL:
	ROI_FILES_MERGE_ALL.append(os.path.join(ROI_DIR, roi_name + '_s0.txt'))

ROI_FILES_MERGE_CORTICAL = []
for roi_name in ROI_NAMES_MERGE_CORTICAL:
	ROI_FILES_MERGE_CORTICAL.append(os.path.join(ROI_DIR, roi_name + '_s0.txt'))
	
def to_file(roi_name):
	return os.path.join(ROI_DIR, roi_name + '_s0.txt')