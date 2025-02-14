#!/bin/bash

#SBATCH --job-name=training
#SBATCH --nodes=1
#SBATCH --exclude=sh00,sh[10-19]
#SBATCH --partition=gpu_prod_long
#SBATCH --time=40:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-1

current_dir=`pwd`
export PATH=$PATH:~/.local/bin

echo "Session " ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "Running on " $(hostname)

echo "Copying the source directory and data"
date
mkdir $TMPDIR/code
rsync -r --exclude logs --exclude logslurms --exclude configs --include ".git/" . $TMPDIR/code

echo "Checking out the correct version of the code commit_id cf5d3f28afabb383c3a06047bf924f90babadcf2"
cd $TMPDIR/code
git checkout cf5d3f28afabb383c3a06047bf924f90babadcf2


echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python3 -m pip install .
python3 -m pip install segmentation_models_pytorch albumentations
python3 -m pip install lion_pytorch
python3 -m pip install monai einops

echo "Training"
python3 torchtmpl/main.py /usr/users/sdim/sdim_22/team-6-kaggle-challenge-deep-learning/configs/tmpy2od95h1-config.yml train

if [[ $? != 0 ]]; then
    exit -1
fi
