#!/bin/sh
runs=33
# 
qsub -t 1-$runs:1 tsgp_f1.sh

#
