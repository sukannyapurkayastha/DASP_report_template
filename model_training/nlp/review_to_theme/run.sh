#!/bin/bash
#
#SBATCH --job-name=attitude_themes
#SBATCH --output=/ukp-storage-1/yang1/EMNLP2023_jiu_jitsu_argumentation_for_rebuttals/codes/review_to_desc/attitude_themes_output.txt
#SBATCH --mail-user=zhijingshui.yang@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_model:a180"
#SBATCH --array=0-9                     # Task array range (10 tasks)

# Activate the Python environment
source /ukp-storage-1/yang1/miniconda3/bin/activate /ukp-storage-1/yang1/miniconda3/envs/jitsupeer

PRETRAINED_MODELS=("bert-base-uncased" "bert-large-uncased" "roberta-base" "roberta-large" "distilbert-base-uncased" \
                   "albert-base-v2" "google/electra-base-discriminator" "t5-small" "t5-base" "t5-large")
SEEDS=(42 123 456 789 321 654 987 111 222 333)

# Get task-specific variables based on SLURM_ARRAY_TASK_ID
PRETRAINED_MODEL=${PRETRAINED_MODELS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Set other parameters
EPOCHS=10
LEARNING_RATE=5e-5

# Run the training script
python train_model.py \
--pretrained_model_path $PRETRAINED_MODEL \
--seed $SEED \
--epochs $EPOCHS \
--learning_rate $LEARNING_RATE