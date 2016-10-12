#!/usr/bin/env bash

# ------------------------------------------------------------------------------------------------------
# SETTINGS

# Specify directory where subject data is located. Within this directory each subject
# is assigned its own subdirectory named according to the subject's ID.
subjects_dir=/data/raw_data/imagemend/uio/smri/output
# Specify subjects file containing subject IDs to process
subjects_file=/data/raw_data/imagemend/uio/smri/subjects_all.txt
# Specify ROI directory
rois_dir=/data/raw_data/rois/harvard-oxford
# Specify ROI file containing ROI names to use
rois_file=/data/raw_data/rois/harvard-oxford/rois_all.txt
# Specify the image file to apply the ROI masks to. The script will automatically
# verify it has a NIFTI extension. Make sure you select the image with the right
# smoothing. 
image_file=nu_GM_to_template_GM_mod_s0.nii.gz
# Specify smoothing factor explicitly
smoothing=0

# ------------------------------------------------------------------------------------------------------
# PREPARATION

# Verify that subjects directory exists.
if [[ ! -d ${subjects_dir} ]] ; then
	echo "[ERROR] Subjects directory ${subjects_dir} does not exist"
	exit 1
fi

# Verify that subjects file exists.
if [[ ! -f ${subjects_file} ]] ; then
	echo "[ERROR] Subjects file ${subjects_file} does not exist"
	exit 1
fi

# Verify that image file has a value
if [[ "${image_file}" == "" ]] ; then
	echo "[ERROR] No image file specified"
	exit 1
fi

# Verify that image file has compressed NIFTI file extension
if [[ ${image_file} == ${image_file/.nii.gz} ]] ; then
	echo "[ERROR] Image file must have *.nii.gz extension"
	exit 1
fi

# Strip extension from image file name
image_file=${image_file/.nii.gz/}

# Create timestamp
timestamp=`date +%Y%m%d_%H%M%S`

# Create log file from timestamp
log_file="${timestamp}.txt"
log_file="${subjects_dir}/${log_file}"
echo "Script: run_vbm.sh" > ${log_file}
message="`date +%Y-%m-%d:%H:%M:%S` [INFO] Log file created"
echo ${message}; 
echo ${message} >> ${log_file}

# # ------------------------------------------------------------------------------------------------------
# # EXTRACT VOXEL COORDINATES AND INTENSITIES

# message="`date +%Y-%m-%d:%H:%M:%S` [INFO] Extracting ROI voxel intensities"
# echo ${message}
# echo ${message} >> ${log_file}
# for roi in `more ${rois_file}` ; do
# 	rm -f runner; touch runner; chmod a+x runner
# 	echo "${FSLDIR}/bin/fslmeants -i ${subjects_dir}/\${1}/${image_file} -m ${rois_dir}/${roi} -o ${subjects_dir}/\${1}/${roi}_s${smoothing}.txt --showall" >> runner
# 	parallel -a ${subjects_file} ./runner
# 	rm -f runner
# 	message="`date +%Y-%m-%d:%H:%M:%S` [INFO] ${roi} - Done"
# 	echo ${message}
# 	echo ${message} >> ${log_file}
# done

# # ------------------------------------------------------------------------------------------------------
# # CREATE CSV FILE WITH VOXEL INTENSITIES FOR EACH ROI

# message="`date +%Y-%m-%d:%H:%M:%S` [INFO] Converting files to CSV format"
# echo ${message}
# echo ${message} >> ${log_file}
# for roi in `more ${rois_file}` ; do

# 	rm -f runner
# 	touch runner
# 	chmod a+x runner

# 	echo "#!/usr/bin/env python" >> runner
# 	echo "import os" >> runner
# 	echo "import sys" >> runner
# 	echo "subject_id = sys.argv[1]" >> runner
# 	echo "file_name = os.path.join('${subjects_dir}', subject_id, '${roi}_s${smoothing}.txt')" >> runner
# 	echo "f = open(file_name, 'r')" >> runner
# 	echo "f.readline()" >> runner
# 	echo "f.readline()" >> runner
# 	echo "f.readline()" >> runner
# 	echo "line = f.readline()" >> runner
# 	echo "f.close()" >> runner
# 	echo "line = subject_id + ',' + line.replace('  ', ',')" >> runner
# 	echo "f = open(file_name, 'w')" >> runner
# 	echo "f.write(line)" >> runner
# 	echo "f.close()" >> runner

# 	parallel -a ${subjects_file} ./runner
# 	rm -f runner

# 	message="`date +%Y-%m-%d:%H:%M:%S` [INFO] ${roi} - Done"
# 	echo ${message}
# 	echo ${message} >> ${log_file}
# done

# ------------------------------------------------------------------------------------------------------
# CREATE ROI FEATURE TABLES

message="`date +%Y-%m-%d:%H:%M:%S` [INFO] Creating ROI feature tables"
echo ${message}
echo ${message} >> ${log_file}

rm -f runner
touch runner
chmod a+x runner

echo "#!/usr/bin/env python" >> runner
echo "import os" >> runner
echo "import sys" >> runner
echo "roi = sys.argv[1]" >> runner
echo "subjects = []" >> runner
echo "f = open('${subjects_file}', 'r')" >> runner
echo "for subject in f.readlines():" >> runner
echo "  subjects.append(subject.strip())" >> runner
echo "f.close()" >> runner
echo "roi_file = os.path.join('${subjects_dir}', roi + '_s${smoothing}.txt')" >> runner
echo "f_roi = open(roi_file, 'w')" >> runner
echo "header = []" >> runner
echo "for subject in subjects:" >> runner
echo "  subject_file = os.path.join('${subjects_dir}', subject, roi + '_s${smoothing}.txt')" >> runner
echo "  f = open(subject_file, 'r')" >> runner
echo "  values = f.readline().strip().split(',')" >> runner
echo "  if len(header) == 0:" >> runner
echo "    header.append('id')" >> runner
echo "    for i in range(len(values[1:])):" >> runner
echo "      header.append('V' + str(i))" >> runner
echo "    f_roi.write(','.join(header) + '\n')" >> runner
echo "  f.close()" >> runner
echo "  f_roi.write(','.join(values) + '\n')" >> runner
echo "f_roi.close()" >> runner
echo "print('{} - Done'.format(roi))" >> runner

parallel -a ${rois_file} ./runner
rm -f runner

# ------------------------------------------------------------------------------------------------------
# FINISH

message="`date +%Y-%m-%d:%H:%M:%S` [INFO] Done"
echo ${message}
echo ${message} >> ${log_file}








