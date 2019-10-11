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
export EXP_NAME="protein_folding"
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
export GPU_DEBUG=0

# Remove previous copy of the repository
rm -rf /scratch/$USER/$EXP_NAME

# Copy the data required
mkdir -p /scratch/$USER/$EXP_NAME/data
cd /scratch/$USER/$EXP_NAME
# rsync -a ada:/share2/$USER/data/casp12/training_30 ./data/proteinnet_raw
# rsync -a ada:/share2/$USER/data/casp12/validation ./data/proteinnet_raw
# rsync -a ada:/share2/$USER/data/casp12/testing ./data/proteinnet_raw
rsync -a ada:/share2/$USER/$EXP_NAME/data/casp12/preprocessed/training_30* ./data/proteinnet_preprocessed
rsync -a ada:/share2/$USER/$EXP_NAME/data/casp12/preprocessed/testing* ./data/proteinnet_preprocessed
rsync -a ada:/share2/$USER/$EXP_NAME/data/casp12/preprocessed/validation* ./data/proteinnet_preprocessed

# Copy the source code
rsync -a zeus@10.2.12.106:/home/zeus/Git/protein-folding-ml/ /scratch/$USER/$EXP_NAME

# Run the necessary commands below
source activate pyt
git checkout raptor-old
# python preprocessing.py
python main.py "$1"
# python testing.py
conda deactivate

rsync -a ./output/ ada:/share2/$USER/outputs/
