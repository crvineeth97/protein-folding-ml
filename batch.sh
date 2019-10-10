#!/bin/bash

##SBATCH --qos=sub
##SBATCH -A sub
##SBATCH -p short
##SBATCH --time=06:00:00
##SBATCH -w gnode02

#SBATCH --qos=medium
#SBATCH -A research
#SBATCH -p long
#SBATCH --time=04-00:00:00
#SBATCH -w gnode06

#SBATCH -n 10
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --mem=0
#SBATCH --mail-type=END
#SBATCH --mail-user=ravindrachelur.v@research.iiit.ac.in

# Set some environment variables for the current script
export EXP_NAME="raptor_contact"
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
export GPU_DEBUG=0

# Copy necessary files to a scratch directory and cd into it
# rm -rf /scratch/$USER/$EXP_NAME
mkdir -p /scratch/$USER/$EXP_NAME
rsync -a ./ /scratch/$USER/$EXP_NAME
cd /scratch/$USER/$EXP_NAME

rsync -a ada:/share2/$USER/data/RaptorX_contact_prediction/pdb25* ./data/raw/

# Run the necessary commands below
source activate pyt
git checkout raptor-old
# python preprocessing.py
python main.py "$1"
# python testing.py
conda deactivate

rsync -a ./ ada:/share2/$USER/$EXP_NAME
