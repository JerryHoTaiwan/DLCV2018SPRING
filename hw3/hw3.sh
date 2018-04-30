#!/bin/bash
wget -O 'fcn_32.h5' 'https://www.dropbox.com/s/pm49qcg1i6sbszn/fcn_32.h5?dl=1'
python3 hw3_baseline.py $1 $2