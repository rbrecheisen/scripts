#!/usr/bin/env bash

# ------------------------------------------------------------------------------------------------------
# SETTINGS

# Specify the directory containing the ROI text files.
rois_dir=${1}
# Specify the output directory for storing the compressed ZIP file
output_dir=${2}
# Specify the output file name (optional)
output_file=${3}

# ------------------------------------------------------------------------------------------------------
# SETTINGS

if [[ ! -d ${rois_dir} ]] ; then
	echo "[ERROR] ROIs directory does not exist"
	exit 1
fi

if [[ ! -d ${output_dir} ]] ; then
	echo "[WARN] Output directory does not exist, will be created"
	mkdir -p ${output_dir}
fi

if [[ "${output_file}" == "" ]] ; then
	output_file=ROIs.tar.gz
fi

# ------------------------------------------------------------------------------------------------------
# RUN FILE COMPRESSION

pwd=`pwd`
cd ${rois_dir}
tar -zcvf ${output_file} GM_*.txt
mv ${output_file} ${output_dir}
cd ${pwd}