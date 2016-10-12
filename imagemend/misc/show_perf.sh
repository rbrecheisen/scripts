#!/bin/bash

dir=/data/raw_data/imagemend/uio/smri/

scp $BI_COMPUTE:${dir}/mem.txt .
scp $BI_COMPUTE:${dir}/disk.txt .

if [ ! -f mem.txt ]; then
	echo "Error: could not retrieve mem.txt"
	exit 1
fi

if [ ! -f disk.txt ]; then
	echo "Error: could not retrieve disk.txt"
	exit 1
fi

python show_mem.py
python show_disk.py
