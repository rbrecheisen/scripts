#!/bin/bash

subj_f=$1
subj_d=$2
f=$3

if [ "${subj_f}" == "" ] || [ "${subj_d}" == "" ] || [ "${f}" == "" ]; then
	echo ""
	echo "Usage: uncompress_nii.sh <subjects file> <subjects dir> <nifti file>"
	echo ""
	echo "  <subjects file> : File containing list of subject IDs."
	echo "  <subjects dir>  : Directory containing subject subdirectories."
	echo "  <nifti file>    : File to uncompress."
	echo ""
	exit 1
fi

for s in `more ${subj_f}`; do
  if [[ $s == \#* ]]; then
  	continue
  fi
  gunzip ${subj_d}/${s}/${f}
done

echo "Done"