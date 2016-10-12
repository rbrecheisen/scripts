#!/bin/bash

subj_f="/Volumes/data/raw_data/imagemend/uio/smri/subjects.csv"
subj_d="/Volumes/data/raw_data/imagemend/uio/smri/raw"
targ_d="/Users/Ralph/datasets/imagemend/data/uio/smri/normalized"
norm_1=swc1nu.nii
norm_2=swc2nu.nii
voxs_1=${norm_1}.txt
voxs_2=${norm_2}.txt

if [ "${subj_f}" == "" ] || [ "${subj_d}" == "" ] || [ "${targ_d}" == "" ]; then
	echo ""
	echo "Usage: copy_normalized.sh <subject file> <subject dir> <target dir>"
	echo ""
	exit 1
fi

for subj in `more ${subj_f}`; do
	
	if [[ ${subj} == \#* ]]; then
		continue
	fi
	
	if [ ! -f ${subj_d}/${subj}/${norm_1} ]; then
		echo "Missing file ${norm_1}"
		continue
	fi	
	
	if [ ! -f ${subj_d}/${subj}/${norm_2} ]; then
		echo "Missing file ${norm_2}"
		continue
	fi

	if [ ! -f ${targ_d}/${subj}/${norm_1} ]; then
		echo "Copying GM for ${subj}"
		mkdir -p ${targ_d}/${subj}
		cp ${subj_d}/${subj}/${norm_1} ${targ_d}/${subj}
	fi

	if [ ! -f ${targ_d}/${subj}/${norm_2} ]; then
		echo "Copying WM for ${subj}"
		mkdir -p ${targ_d}/${subj}
		cp ${subj_d}/${subj}/${norm_2} ${targ_d}/${subj}
	fi

	if [ ! -f ${targ_d}/${subj}/${voxs_1} ]; then
		echo "Copying exported GM voxels for ${subj}"
		mkdir -p ${targ_d}/${subj}
		cp ${subj_d}/${subj}/${voxs_1} ${targ_d}/${subj}
	fi

	if [ ! -f ${targ_d}/${subj}/${voxs_2} ]; then
		echo "Copying exported WM voxels for ${subj}"
		mkdir -p ${targ_d}/${subj}
		cp ${subj_d}/${subj}/${voxs_2} ${targ_d}/${subj}
	fi

done

echo "Done"
