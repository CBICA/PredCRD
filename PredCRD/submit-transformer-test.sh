#!/bin/bash

######################## START OF SLURM COMMANDS ########################
# vim: ft=slurm
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-gpu=16
#SBATCH --job-name=traing_transformer
#SBATCH --output ../log/ROI_SurrealGAN-%j.out
#SBATCH --partition=all
#SBATCH --time=1-23:00:00
######################### END OF SLURM COMMANDS #########################

source activate /cbica/home/baikk/.conda/envs/PredCRD_env

python3 surrealGAN_tabular_inference.py