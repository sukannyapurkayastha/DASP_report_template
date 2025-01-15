---
library_name: transformers
license: apache-2.0
base_model: t5-large
tags:
- generated_from_trainer
metrics:
- rouge
model-index:
- name: 1e-4
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# 1e-4

This model is a fine-tuned version of [t5-large](https://huggingface.co/t5-large) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4278
- Rouge1: 44.4982
- Rouge2: 35.3017
- Rougel: 43.8864
- Rougelsum: 44.1202
- Gen Len: 13.6791

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 10.0

### Training results



### Framework versions

- Transformers 4.46.3
- Pytorch 2.4.1+cu121
- Datasets 3.1.0
- Tokenizers 0.20.3
