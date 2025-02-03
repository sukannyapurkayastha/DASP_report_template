#!/bin/bash
#
#SBATCH --job-name=traint5
#SBATCH --output=/ukp-storage-1/oehler/refactoring/res.txt
#SBATCH --mail-user=philipp.oehler@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=256GB
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu_model:h100pcie"


source /ukp-storage-1/oehler/miniconda3/bin/activate /ukp-storage-1/oehler/miniconda3/envs/summary_env6


srun python /ukp-storage-1/oehler/refactoring/main.py
