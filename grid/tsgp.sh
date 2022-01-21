#$ -S /bin/sh
#$ -r y
#
# I know I have a directory here so I'll use it as my initial working directory
#
#$ -wd /vol/grid-solar/sgeusers/gaoyuan7
#
# End of the setup directives
#
# Now let's do something useful, but first change into the job-specific
# directory that should have been created for us
#
# Check we have somewhere to work now and if we don't, exit nicely.
#
if [ -d /local/tmp/gaoyuan7/$JOB_ID.$SGE_TASK_ID ]; then
        cd /local/tmp/gaoyuan7/$JOB_ID.$SGE_TASK_ID
else
        echo "Uh oh ! There's no job directory to change into "
        echo "Something is broken. I should inform the programmers"
        echo "Save some information that may be of use to them"
        echo "Here's LOCAL TMP "
        ls -la /local/tmp
        echo "AND LOCAL TMP gaoyuan7 "
        ls -la /local/tmp/gaoyuan7
        echo "Exiting"
        exit 1
fi
#
echo $1
echo $2
echo $SGE_TASK_ID
# Now we are in the job-specific directory so now can do something useful
##file path
file_path=/vol/grid-solar/sgeusers/gaoyuan7/gp1/

# main algorithm name
# Copy the input file to the local directory
# datasets
cp -arv /vol/grid-solar/sgeusers/gaoyuan7/gp1/datasets/$1/* .
# deap master.

cp -arv $file_path/algorithms/* .
# Stdout from programs and shell echos will go into the file
#    scriptname.o$JOB_ID
#  so we'll put a few things in there to help us see what went on
#
#
python3 MLGP_Main.py $SGE_TASK_ID $1

cp *.txt  $file_path/results/MLGP/$1
cp *.log  $file_path/results/MLGP/$1
cp *.pickle  $file_path/results/MLGP/$1

echo "Ran through OK"

