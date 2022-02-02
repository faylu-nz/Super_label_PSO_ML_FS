#!/bin/sh

#SBATCH --job-name=PSO_HL
#SBATCH --output=out_array_%A_%a.out
#SBATCH --error=out_array_%A_%a.err
#SBATCH -a 1-30
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-0:00:00
#SBATCH --partition=parallel

hosts=$SLURM_JOB_NODELIST
homeDir='/nfs/home/luyan1/PSO_raapoi
cd $homeDir/code/

# 1-dataset
outDir=$homeDir/Results/$1/
inDir='/nfs/home/nguyenba/Dataset/'
mkdir -p $outDir

python -W ignore Main.py $1 ${SLURM_ARRAY_TASK_ID} $inDir $outDir 'not_parallel' $2 $3


#move the error and output file to tmp folder
mkdir -p $homeDir/Out/
mv *.out $homeDir/Out/
mkdir -p $homeDir/Err/
mv *.err $homeDir/Err/

