#!/bin/bash -l
#
#SBATCH --ntasks=1
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=trainsnn_0.01
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

WORKDIR="$TMPDIR/$SLURM_JOBID" 
mkdir "$WORKDIR"
cd "$WORKDIR"

for f in $WORK/CXR8/images/*.tar.gz; do tar xf "$f"; done
echo All images has been successfully extracted.

export https_proxy=http://proxy:80
module load python
source activate MasterThesis

python3 /home/woody/iwi5/iwi5155h/ExperimentClassifierFreeze/Experiment_0.01/test/retrain_SNN.py --config_path /home/woody/iwi5/iwi5155h/ExperimentClassifierFreeze/Experiment_0.01/test/config_files/ --config config_retrainSNN.json --images_path $WORKDIR/images/

conda deactivate
module unload python
cd; rm -r "$WORKDIR"