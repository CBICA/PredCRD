#!/bin/bash

######################## START OF SLURM COMMANDS ########################
# vim: ft=slurm
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gpus=p100:1
#SBATCH --cpus-per-gpu=16
#SBATCH --job-name=traing_transformer
#SBATCH --output ../log/ROI_SurrealGAN-j.log 
#SBATCH --partition=all
#SBATCH --time=1-23:00:00
######################### END OF SLURM COMMANDS #########################

source activate /cbica/home/wudi2/.conda/envs/Diii

python surrealGAN_tabular_prediction.py