#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p long
#SBATCH -n 10
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=MaxMemPerCPU
#SBATCH --mem=0
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=ravindrachelur.v@research.iiit.ac.in

# Set some environment variables for the current script
export EXP_NAME="openprotein"

# Copy necessary files to a scratch directory and cd into it
rsync -avP ./ /scratch/$USER/$EXP_NAME
cd /scratch/$USER/$EXP_NAME

# Load the necessary modules
module load cuda/9.0
module load cudnn/7-cuda-9.0
module load openmpi/2.1.1-cuda9

# Run the necessary commands below


# Copy the files in /scratch to /share2
rsync -avP ./ ada:/share2/crvineeth97/$EXP_NAME