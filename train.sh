#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p long
#SBATCH -n 40
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=4096
#SBATCH --mem=0
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=ravindrachelur.v@research.iiit.ac.in

# Set some environment variables for the current script
export EXP_NAME="protein_folding"

# Copy necessary files to a scratch directory and cd into it
mkdir -p /scratch/$USER/$EXP_NAME
rsync -avP ./ /scratch/$USER/$EXP_NAME
cd /scratch/$USER/$EXP_NAME

rsync -avP ada:/share2/$USER/$EXP_NAME/data/preprocessed ./data

# Load the necessary modules
module add cuda/10.0
module add cudnn/7-cuda-10.0

# Run the necessary commands below
mkdir -p output/models
source activate pyt
python3 main.py --hide-ui
conda deactivate pyt

rsync -avP ./ ada:/share2/$USER/$EXP_NAME
