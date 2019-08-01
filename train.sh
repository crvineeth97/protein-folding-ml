#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p long
#SBATCH -n 10
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --mem=0
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=ravindrachelur.v@research.iiit.ac.in

###SBATCH -w gnode32
# Set some environment variables for the current script
export EXP_NAME="dihedral_predictions"

# Copy necessary files to a scratch directory and cd into it
mkdir -p /scratch/$USER/$EXP_NAME
rsync -qavP ./ /scratch/$USER/$EXP_NAME
cd /scratch/$USER/$EXP_NAME

# rsync -qavP ada:/share2/$USER/data/casp12/training_30 ./data/raw/
# rsync -qavP ada:/share2/$USER/data/casp12/validation ./data/raw/
# rsync -qavP ada:/share2/$USER/data/casp12/testing ./data/raw/
rsync -qavP ada:/share2/$USER/$EXP_NAME/data/preprocessed ./data

# Load the necessary modules
module load cuda/10.0
module load cudnn/7-cuda-10.0

# Run the necessary commands below
source activate pyt
python main.py
conda deactivate pyt

rsync -qavP ./ ada:/share2/$USER/$EXP_NAME
