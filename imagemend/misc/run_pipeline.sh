#!/usr/bin/env bash

# Specify directory containing subject images. For each subject, there must be
# a subdirectory named according to the subject's ID. The script will automatically
# find each subject's subdirectory.
subjects_dir=/data/raw_data/imagemend/uio/smri
# Specify path to subjects file containing subject IDs
subjects_file=/data/raw_data/imagemend/uio/smri/subjects.txt
# Specify directory where subject data is located. Within this directory each subject
# is assigned its own subdirectory named according to the subject's ID.
output_dir=/data/raw_data/imagemend/uio/smri/output
# Specify ROI directory
rois_dir=/data/raw_data/rois/harvard-oxford
# Specify ROI file containing ROI names to use
rois_file=/data/raw_data/rois/harvard-oxford/rois.txt
# Specify the image file to start with. The pipeline will automatically detect the
# extension. If the extension is *.mgz then it will convert to *.nii.gz first.
image_file1=nu.nii.gz
# Specify the image file to apply the ROI masks to. The script will automatically
# verify it has a NIFTI extension. Make sure you select the image with the right
# smoothing. 
image_file2=nu_GM_to_template_GM_mod_s0.nii.gz
# Specify smoothing factor explicitly so we can append it to the ROI output files.
# If we want to re-run the script for different smoothings we get additional files.
smoothing=0
# Specify directory for storing compressed ZIP file
zip_dir=/data/raw_data/imagemend/uio
# Specify name of ZIP file
zip_file=smri_ROIs.tar.gz

# ------------------------------------------------------------------------------------------------------

# # Run VBM alignment
# vbm.sh ${subjects_dir} ${subjects_file} ${image_file1}

# Run ROI extraction
roi_extract.sh ${output_dir} ${subjects_file} ${rois_dir} ${rois_file} ${image_file2} ${smoothing}

# # Compress ROI files
# roi_zip.sh ${output_dir} ${zip_dir} ${zip_file}






