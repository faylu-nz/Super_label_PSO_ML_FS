#!/bin/bash

#SBATCH --partition=parallel


datasets=( 'Vehicle' 'ImageSegmentation' 'WallRobot' 'German' 'WBCD' 'GuesterPhase' 'Dermatology' 'Ionosphere' 'Chess' 'Sonar' 'Plant' 'Mice' 'Movementlibras' 'Hillvalley' 'Musk1' 'Semeion' 'LSVT' 'Madelon' 'Isolet' 'MultipleFeatures' 'Gametes' 'QsarAndrogenReceptor' 'COIL20' 'ORL' 'Yale' 'Bioresponse' 'Colon' 'SRBCT' 'warpAR10P' 'PCMAC' 'RELATHE' 'BASEHOCK' 'Prostate-GE' 'Carcinom' 'Ovarian' 'GLI-85' )
datasets=( 'Vehicle' )
for dataset in "${datasets[@]}"
do
	echo $dataset
	echo '----'$dataset
	#sbatch Job.sh $dataset 'constrained-single-fit' 'local'
	#sbatch Job.sh $dataset 'constrained-single-fit' 'not-local'
	sbatch Job.sh $dataset 'not-constrained' 'local'
	sbatch Job.sh $dataset 'not-constrained' 'not-local'
done

datasets=( 'USPS' )
datasets=( )
for dataset in "${datasets[@]}"
do
	echo $dataset
	echo '----'$dataset
	#sbatch Job_parallel.sh $dataset 'constrained-single-fit' 'local'
	#sbatch Job_parallel.sh $dataset 'constrained-single-fit' 'not-local'
	sbatch Job_parallel.sh $dataset 'not-constrained' 'local'
	sbatch Job_parallel.sh $dataset 'not-constrained' 'not-local'
done


