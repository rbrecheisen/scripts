#!/bin/bash

subj_f=$1
subj_d=$2
f_keep=$3

if [ "${subj_f}" == "" ] || [ "${subj_d}" == "" ] || [ "${f_keep}" == "" ]; then
	echo ""
	echo "Usage: clean_spm.sh <subject file> <subject dir> <file to keep>"
	echo ""
	echo "  <subject file> : File containing list of subject IDs."
	echo "  <subject dir>  : Folder containing subject subdirectories."
	echo "  <feel to keep> : File name to keep."
	echo ""
	exit 1
fi

for subj in `more ${subj_f}`; do

	if [[ ${subj} == \#* ]]; then
		continue
	fi

	if [ ! -f ${subj_d}/${subj}/*.mat ]; then
		echo "Nothing to clean for ${subj}"
		continue
	fi

	# Backup nu.nii
	mkdir -p ${subj_d}/${subj}/tmp
	mv ${subj_d}/${subj}/${f_keep} ${subj_d}/${subj}/tmp

	# Remove everything else
	rm -f ${subj_d}/${subj}/*.nii
	rm -f ${subj_d}/${subj}/*.mat

	# Restore nu.nii
	mv ${subj_d}/${subj}/tmp/${f_keep} ${subj_d}/${subj}
	rmdir ${subj_d}/${subj}/tmp

done
