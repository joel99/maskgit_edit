#!/bin/bash
#SBATCH --job-name=16726_final
#SBATCH --gres gpu:1
#SBATCH -p gpu
#SBATCH -c 6
#SBATCH -t 1:00:00
#SBATCH --mem 30G
#SBATCH --output=slurm_logs/%j.out

echo $@
hostname
source ~/.bashrc # Note bashrc has been modified to allow slurm jobs
conda activate jax4
python -u eval_batch.py $@
