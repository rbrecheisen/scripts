#!/bin/bash

sf=$1          # Subjects file.
sd=$2          # Subjects directory.
tf=$3          # Target file.
of=$4          # Output file.

if [ "${sf}" == "" ]; then
	echo ""
	echo "Usage: zip_files <subjects file> <subjects dir> <nifti file> <output file>"
	echo ""
	echo "  <subjects file> : File containing list of subject IDs."
	echo "  <subjects dir>  : Directory containing subject subdirectories."
	echo "  <target file>   : Name of file to compress."
	echo "  <output file>   : Output .tar.gz file."
	echo ""
	exit 1
fi

rm -f 'files.txt'
for s in `more ${sf}`; do
	if [[ ${s} == \#* ]]; then
		continue
	fi
	echo ${sd}/${s}/${tf} >> 'files.txt'
done

tar cvf ${of} -T 'files.txt'
rm -f 'files.txt'
echo "done"
