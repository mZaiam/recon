#!/bin/bash
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -p batch-AMD
#SBATCH --time=120:00:00

source ~/.bashrc

conda activate ai

python -u processing.py
