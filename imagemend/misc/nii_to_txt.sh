#!/bin/bash

subj_f="/data/raw_data/imagemend/uio/smri/subjects.csv"
subj_d="/data/raw_data/imagemend/uio/smri/raw"
file_1="swc1nu.nii"
file_2="swc2nu.nii"

rm -f runner; touch runner; chmod a+x runner
echo "if [[ \${1} == \\#* ]]; then" >> runner
echo "  exit 1" >> runner
echo "fi" >> runner
echo "echo \"Exporting voxels for subject \${1}\"" >> runner
echo "fslmeants -i ${subj_d}/\${1}/${file_1} -o ${subj_d}/\${1}/${file_1}.txt --showall" >> runner
echo "fslmeants -i ${subj_d}/\${1}/${file_2} -o ${subj_d}/\${1}/${file_2}.txt --showall" >> runner
parallel -a ${subj_f} ./runner
rm -f runner

echo "Done"
