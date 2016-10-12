#!/bin/bash

rm -f *.m~
rm -f *.pyc

TARGET='ralph@192.168.26.103:/data/software/scripts'
rsync -av --delete * $TARGET
