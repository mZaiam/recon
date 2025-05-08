#!/bin/bash
#SBATCH -n 8
#SBATCH --ntasks-per-node=8
#SBATCH -p gpu
#SBATCH --gres=gpu:4090:1
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate ai
module load cuda/12.1
python -u train_ae.py --ld=$1 
