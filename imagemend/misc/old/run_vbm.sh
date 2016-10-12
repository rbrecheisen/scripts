#!/usr/bin/env bash

# ------------------------------------------------------------------------------------------------------
# SETTINGS

# Specify prior image. This will be used for the initial registration to MNI.
prior=${FSLDIR}/data/standard/tissuepriors/avg152T1_gray
# Specify directory containing subject images. For each subject, there must be
# a subdirectory named according to the subject's ID. The script will automatically
# find each subject's subdirectory.
subjects_dir=/data/raw_data/imagemend/uio/smri
# Specify path to subjects file containing subject IDs
subjects_file=/data/raw_data/imagemend/uio/smri/subjects_all.txt
# Specify the image file to start with. The pipeline will automatically detect the
# extension. If the extension is *.mgz then it will convert to *.nii.gz first.
image_file=nu.nii.gz

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

# Check if output directory exists. If so, show a warning that it will
# be deleted unless the user moves it somewhere else.
output_dir=${subjects_dir}/output
if [[ -d ${output_dir} ]] ; then
	echo ""
	echo "[WARN] Output directory exists."
	echo "[WARN] Press Ctrl+C to quit and move the directory somewhere else."
	echo "[WARN] Press any key to delete the output directory and continue."
	read -n 1; echo
fi
rm -rf ${output_dir}

# Create log file from timestamp
log_file="${timestamp}.txt"
log_file="${subjects_dir}/${log_file}"
echo "Script: run_vbm.sh" > ${log_file}
message="`date +%Y-%m-%d:%H:%M:%S` [INFO] Log file created"; 
echo ${message}; 
echo ${message} >> ${log_file}

# Create subject directories in output directory and copy original image
# to the output directory.
message="`date +%Y-%m-%d:%H:%M:%S` [INFO] Initializing output directory"
echo ${message}
echo ${message} >> ${log_file}
mkdir ${output_dir}
for subject in `more ${subjects_file}` ; do
	mkdir -p ${output_dir}/${subject}
	cp ${subjects_dir}/${subject}/${image_file}.nii.gz ${output_dir}/${subject}
done

# ------------------------------------------------------------------------------------------------------
# REORIENTATION TO STANDARD SPACE

message="`date +%Y-%m-%d:%H:%M:%S` [INFO] Reorienting images to standard space"
echo ${message}
echo ${message} >> ${log_file}
rm -f runner; touch runner; chmod a+x runner
echo "${FSLDIR}/bin/fslreorient2std ${output_dir}/\${1}/${image_file} ${output_dir}/\${1}/${image_file}_reorient" >> runner
parallel -a ${subjects_file} ./runner
rm -f runner

# ------------------------------------------------------------------------------------------------------
# BRAIN EXTRACTION

message="`date +%Y-%m-%d:%H:%M:%S` [INFO] Extracting brains"
echo ${message}
echo ${message} >> ${log_file}
rm -f runner; touch runner; chmod a+x runner
echo "${FSLDIR}/bin/bet ${output_dir}/\${1}/${image_file}_reorient ${output_dir}/\${1}/${image_file}_brain" >> runner
echo "${FSLDIR}/bin/standard_space_roi ${output_dir}/\${1}/${image_file}_brain ${output_dir}/\${1}/${image_file}_cut -roiNONE -ssref ${FSLDIR}/data/standard/MNI152_T1_2mm_brain -altinput ${output_dir}/\${1}/${image_file}_reorient" >> runner
echo "${FSLDIR}/bin/bet ${output_dir}/\${1}/${image_file}_cut ${output_dir}/\${1}/${image_file}_brain -f 0.4 -R" >> runner
parallel -a ${subjects_file} ./runner
rm -f runner

# ------------------------------------------------------------------------------------------------------
# GRAY MATTER SEGMENTATION

message="`date +%Y-%m-%d:%H:%M:%S` [INFO] Segmenting gray and white matter"
echo ${message}
echo ${message} >> ${log_file}
rm -f runner; touch runner; chmod a+x runner
echo "${FSLDIR}/bin/fast -R 0.3 -H 0.1 ${output_dir}/\${1}/${image_file}_brain" >> runner
echo "${FSLDIR}/bin/imcp ${output_dir}/\${1}/${image_file}_brain_pve_1 ${output_dir}/\${1}/${image_file}_GM" >> runner
parallel -a ${subjects_file} ./runner
rm -f runner

# ------------------------------------------------------------------------------------------------------
# REGISTRATION TO SPATIAL PRIOR

message="`date +%Y-%m-%d:%H:%M:%S` [INFO] Registering images to spatial prior"
echo ${message}
echo ${message} >> ${log_file}
rm -f runner; touch runner; chmod a+x runner
echo "${FSLDIR}/bin/fsl_reg ${output_dir}/\${1}/${image_file}_GM ${prior} ${output_dir}/\${1}/${image_file}_GM_to_prior -a" >> runner
parallel -a ${subjects_file} ./runner
rm -f runner

# ------------------------------------------------------------------------------------------------------
# CREATE INITIAL TEMPLATE

message="`date +%Y-%m-%d:%H:%M:%S` [INFO] Creating initial template"
echo ${message}
echo ${message} >> ${log_file}
image_list=""
for subject in `more ${subjects_file}` ; do
	image_list="${image_list} ${output_dir}/${subject}/${image_file}_GM_to_prior"
done
${FSLDIR}/bin/fslmerge -t ${output_dir}/template_4D_GM ${image_list}
${FSLDIR}/bin/fslmaths ${output_dir}/template_4D_GM -Tmean ${output_dir}/template_GM
${FSLDIR}/bin/fslswapdim ${output_dir}/template_GM -x y z ${output_dir}/template_GM_flipped
${FSLDIR}/bin/fslmaths ${output_dir}/template_GM -add ${output_dir}/template_GM_flipped -div 2 ${output_dir}/template_GM_init

# ------------------------------------------------------------------------------------------------------
# REGISTRATION TO INITIAL TEMPLATE

message="`date +%Y-%m-%d:%H:%M:%S` [INFO] Registering images to initial template"
echo ${message}
echo ${message} >> ${log_file}
rm -f runner; touch runner; chmod a+x runner
echo "${FSLDIR}/bin/fsl_reg ${output_dir}/\${1}/${image_file}_GM ${output_dir}/template_GM_init ${output_dir}/\${1}/${image_file}_GM_to_prior_init -fnirt \"--config=GM_2_MNI152GM_2mm.cnf\"" >> runner
parallel -a ${subjects_file} ./runner
rm -f runner

# ------------------------------------------------------------------------------------------------------
# CREATE FINAL TEMPLATE

message="`date +%Y-%m-%d:%H:%M:%S` [INFO] Create final template"
echo ${message}
echo ${message} >> ${log_file}
image_list=""
for subject in `more ${subjects_file}` ; do
	image_list="${image_list} ${output_dir}/${subject}/${image_file}_GM_to_prior_init"
done
${FSLDIR}/bin/fslmerge -t ${output_dir}/template_4D_GM ${image_list}
${FSLDIR}/bin/fslmaths ${output_dir}/template_4D_GM -Tmean ${output_dir}/template_GM
${FSLDIR}/bin/fslswapdim ${output_dir}/template_GM -x y z ${output_dir}/template_GM_flipped
${FSLDIR}/bin/fslmaths ${output_dir}/template_GM -add ${output_dir}/template_GM_flipped -div 2 ${output_dir}/template_GM_final

# ------------------------------------------------------------------------------------------------------
# REGISTRATION TO FINAL TEMPLATE

message="`date +%Y-%m-%d:%H:%M:%S` [INFO] Registering images to final template"
echo ${message}
echo ${message} >> ${log_file}
rm -f runner; touch runner; chmod a+x runner
echo "${FSLDIR}/bin/fsl_reg ${output_dir}/\${1}/${image_file}_GM ${output_dir}/template_GM_final ${output_dir}/\${1}/${image_file}_GM_to_template_GM -fnirt \"--config=GM_2_MNI152GM_2mm.cnf --jout=${output_dir}/\${1}/${image_file}_JAC_nl\"" >> runner
parallel -a ${subjects_file} ./runner
rm -f runner

# ------------------------------------------------------------------------------------------------------
# MODULATION

message="`date +%Y-%m-%d:%H:%M:%S` [INFO] Modulating images using Jacobian determinants"
echo ${message}
echo ${message} >> ${log_file}
rm -f runner; touch runner; chmod a+x runner
echo "${FSLDIR}/bin/fslmaths ${output_dir}/\${1}/${image_file}_GM_to_template_GM -mul ${output_dir}/\${1}/${image_file}_JAC_nl ${output_dir}/\${1}/${image_file}_GM_to_template_GM_mod_s0 -odt float" >> runner
parallel -a ${subjects_file} ./runner
rm -f runner

# ------------------------------------------------------------------------------------------------------
# SMOOTHING

message="`date +%Y-%m-%d:%H:%M:%S` [INFO] Smoothing images with 1, 3, and 5 mm kernels"
echo ${message}
echo ${message} >> ${log_file}
rm -f runner; touch runner; chmod a+x runner
echo "${FSLDIR}/bin/fslmaths ${output_dir}/\${2}/${image_file}_GM_to_template_GM_mod_s0 -s \${1} ${output_dir}/\${2}/${image_file}_GM_to_template_GM_mod_s\${1}" >> runner
parallel -a ${subjects_file} ./runner 1
parallel -a ${subjects_file} ./runner 3
parallel -a ${subjects_file} ./runner 5
rm -f runner

# ------------------------------------------------------------------------------------------------------
# FINISH

message="`date +%Y-%m-%d:%H:%M:%S` [INFO] Done"
echo ${message}
echo ${message} >> ${log_file}
















