#!/bin/bash

#SBATCH --job-name=templatecode
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=4:00:00
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
rsync -r --exclude logs --exclude logslurms --exclude configs . $TMPDIR/code

echo "Checking out the correct version of the code commit_id efa7bc69bcf3aed255c6730cfb72cb419690eb7e"
cd $TMPDIR/code
git checkout efa7bc69bcf3aed255c6730cfb72cb419690eb7e


echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python3 -m pip install .

echo "Training"
python3 torchtmpl/main.py /usr/users/sdim/sdim_22/team-6-kaggle-challenge-deep-learning/pytorch_template_code-main/configs/tmp0rb5vtrv-config.yml train

if [[ $? != 0 ]]; then
    exit -1
fi
