#!/bin/bash

#SBATCH --qos=sub
#SBATCH -A sub
#SBATCH -p short
#SBATCH --time=06:00:00
#SBATCH -w gnode02

##SBATCH --qos=medium
##SBATCH -A research
##SBATCH -p long
##SBATCH --time=04-00:00:00
##SBATCH -w gnode15

#SBATCH -n 10
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --mem=0
#SBATCH --mail-type=END
#SBATCH --mail-user=ravindrachelur.v@research.iiit.ac.in

# Set some environment variables for the current script
export EXP_NAME="dihedral_predictions"

# Copy necessary files to a scratch directory and cd into it
# rm -rf /scratch/$USER/$EXP_NAME
mkdir -p /scratch/$USER/$EXP_NAME
rsync -a ./ /scratch/$USER/$EXP_NAME
cd /scratch/$USER/$EXP_NAME

# rsync -a ada:/share2/$USER/data/casp12/training_30 ./data/raw/
# rsync -a ada:/share2/$USER/data/casp12/validation ./data/raw/
# rsync -a ada:/share2/$USER/data/casp12/testing ./data/raw/
rsync -a ada:/share2/$USER/$EXP_NAME/data/preprocessed/training_30* ./data/preprocessed
rsync -a ada:/share2/$USER/$EXP_NAME/data/preprocessed/testing* ./data/preprocessed
rsync -a ada:/share2/$USER/$EXP_NAME/data/preprocessed/validation* ./data/preprocessed

# Run the necessary commands below
conda activate pyt
# python preprocessing.py
python main.py "$1"
# python testing.py
conda deactivate

rsync -a ./ ada:/share2/$USER/$EXP_NAME
