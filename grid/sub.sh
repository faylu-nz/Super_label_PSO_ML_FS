#!/bin/sh
runs=33
# 
dataset1='Sprat'
qsub -t 1-$runs:1 tsgp.sh $dataset1 
#
dataset1='Gilt'
qsub -t 1-$runs:1 tsgp.sh $dataset1
#
dataset1='Hourse'
qsub -t 1-$runs:1 tsgp.sh $dataset1
#
