#!/bin/bash

Usage() {
	echo ""
	echo "Usage: to_nifti.sh --subj_file <subj_file> --subj_dir <subj_dir>"
	echo ""
	echo "    <subj_file> : File containing list of subject IDs"
	echo "    <subj_dir>  : Directory containing subject folders"
	echo ""
	exit 1
}

[[ "${1}" = "" ]] && Usage

export FREESURFER_HOME=.

subj_file=${2}
subj_dir=${4}

rm -f convert.sh
echo "./mri_convert \${1}/\${2}/mri/nu.mgz \${1}/\${2}/nu.nii.gz" >> convert.sh
chmod a+x convert.sh
parallel -a ${subj_file} ./convert.sh ${subj_dir}
rm convert.sh
