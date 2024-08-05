#!/bin/bash -l
#
#SBATCH --ntasks=1
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --job-name=trainArchitect_0.008
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
export CUDA_LAUNCH_BLOCKING=1


WORKDIR="$TMPDIR/$SLURM_JOBID" 
mkdir "$WORKDIR"
cd "$WORKDIR"

for f in $WORK/CXR8/images/*.tar.gz; do tar xf "$f"; done
echo All images has been successfully extracted.

export https_proxy=http://proxy:80
module load python
source activate MasterThesis

python3 /home/woody/iwi5/iwi5155h/GaussianNoise/Noise_0.008/test/train_architecture.py --config_path /home/woody/iwi5/iwi5155h/GaussianNoise/Noise_0.008/test/config_files/ --config config_anonymization.json --images_path $WORKDIR/images/

conda deactivate
module unload python
cd; rm -r "$WORKDIR"
 