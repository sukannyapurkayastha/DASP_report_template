#!/bin/bash
#
#SBATCH --job-name=review_to_desc
#SBATCH --output=/ukp-storage-1/yang1/EMNLP2023_jiu_jitsu_argumentation_for_rebuttals/codes/review_to_desc/reviews_to_desc_output.txt
#SBATCH --mail-user=zhijingshui.yang@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_model:a180"

# Activate the Python environment
source /ukp-storage-1/yang1/miniconda3/bin/activate /ukp-storage-1/yang1/miniconda3/envs/jitsupeer

epochs=10
learning_rate=1e-4

srun python main.py \
--model_name_or_path t5-large \
--do_train \
--do_eval \
--do_predict \
--num_beams 5 \
--train_file  'data/train.csv' \
--validation_file 'data/dev.csv' \
--test_file  'data/test.csv' \
--output_dir t5-large-output/$epochs/$learning_rate \
--overwrite_output_dir \
--per_device_train_batch_size=8 \
 --per_device_eval_batch_size=8 \
--gradient_accumulation_steps 4 \
--num_train_epochs $epochs \
--learning_rate $learning_rate \
--save_steps -1 \
 --report_to 'none' \
--predict_with_generate
