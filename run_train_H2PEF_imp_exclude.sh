#!/bin/bash
#SBATCH --job-name=ecg_training            # Job name
#SBATCH --partition=euan                   # Partition to submit to
#SBATCH --output=/oak/stanford/groups/euan/projects/jackos/ecg/files_from_git_AI_ECG_HFPEF/output/train_output_H2PEF_imp_excludes.out  # Standard output file
#SBATCH --error=/oak/stanford/groups/euan/projects/jackos/ecg/files_from_git_AI_ECG_HFPEF/output/train_error_H2PEF_imp_excludes.err    # Standard error file
#SBATCH --time=2-00:00:00                  # Time limit (2 days)
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of tasks (1)
#SBATCH --cpus-per-task=4                  # Number of CPU cores per task
#SBATCH --mem=80G                          # Memory per node
#SBATCH --gres=gpu:1                       # Request 1 GPU (adjust number if needed)

# Activate the Conda environment
source /home/users/jackos/conda_envs/ECG_new/bin/activate

# Run your Python script
python /oak/stanford/groups/euan/projects/jackos/ecg/git/AI_ECG_HFPEF/ecg.py --mode train --task H2PEF_imp_excludes

