#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=test_gpu
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --gpus=1  # Request 1 GPU

module load cuda
srun hostname
