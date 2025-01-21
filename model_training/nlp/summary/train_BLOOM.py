"""
train_BLOOM.py

This module provides functionality to train a BLOOM model (bigscience/bloom-1b1) 
with a BERT-Score metric, using the Hugging Face `Trainer`, 
while masking the prompt portion so the model only learns to predict the output.
"""

from typing import List, Dict

import logging
import os
import sys
import json
import pandas as pd
import torch

import evaluate  # BERT-Score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# Local imports from your project
from data.load_data import load_jsonl
from compute_metrics import compute_metrics

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.CRITICAL,  # Set to DEBUG for detailed logs
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Instantiate BERT-Score metric (optional if you have your own metric)
bertscore_metric = evaluate.load("bertscore")


def prepare_dataset(
    data: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    max_input_length: int = 1024,
    max_output_length: int = 150
) -> Dataset:
    """
    Converts raw data into a Hugging Face Dataset and tokenizes it for BLOOM.
    Masks out the prompt tokens in the labels, so the model only predicts the output portion.

    Args:
        data (List[Dict[str, str]]): A list of dictionaries containing 'input' and 'output' text.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the BLOOM model.
        max_input_length (int, optional): Maximum token length for the input prompts. Defaults to 512.
        max_output_length (int, optional): Maximum token length for the output texts. Defaults to 512.

    Returns:
        Dataset: A tokenized Hugging Face Dataset ready for training and evaluation.
    """
    logger.info("Converting data to a Hugging Face Dataset")
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)

    logger.info("Tokenizing data")

    def tokenize_single_example(example):
        """
        Tokenizes a single example by combining the input prompt with the output text,
        then masks the prompt tokens in the labels.

        Args:
            example (Dict[str, str]): A dictionary with 'input' and 'output' keys.

        Returns:
            Dict[str, List[int]]: A dictionary containing 'input_ids', 'attention_mask', and 'labels'.
        """
        # 1) Build the textual prompt
        prompt_text = example["input"]  # As we already make loaded data a prompt we no longer need this. build_prompt(example["input"])

        # 2) Concatenate prompt + "\n" + output
        full_text = f"{prompt_text}\n{example['output']}"

        # 3) Tokenize the entire sequence
        encoded_full = tokenizer(
            full_text,
            max_length=(max_input_length + max_output_length),
            truncation=True,
            padding="max_length"
        )

        # 3a) Also tokenize just the prompt so we know how many tokens belong to the prompt
        encoded_prompt = tokenizer(
            prompt_text,
            max_length=max_input_length,
            truncation=True,
            padding="max_length"
        )

        # Count how many real (non-padding) tokens are in the prompt
        # (i.e., ignore pad tokens so we know the “true” prompt length)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        prompt_len = sum(1 for t in encoded_prompt["input_ids"] if t != pad_id)

        # 4) Copy the input_ids to labels
        labels = encoded_full["input_ids"].copy()

        # Mask out the prompt tokens in the label
        # This ensures the model is only “penalized” for the output portion
        for i in range(prompt_len):
            labels[i] = -100

        # Return the dictionary in the format the Trainer expects
        return {
            "input_ids": encoded_full["input_ids"],
            "attention_mask": encoded_full["attention_mask"],
            "labels": labels
        }

    # Apply our function example-by-example (batched=False)
    tokenized_dataset = dataset.map(tokenize_single_example, batched=False)
    return tokenized_dataset


def train_bloom_model(
    train_data: List[Dict[str, str]],
    val_data: List[Dict[str, str]],
    model_name: str = "bigscience/bloom-1b1",
    output_dir: str = "./models/bloom",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    learning_rate: float = 5e-5,
    gradient_accumulation_steps: int = 1,
    early_stopping_patience: int = 2
):
    """
    Train a BLOOM model using the provided training and validation data,
    masking the prompt so only the output portion is predicted.
    Uses BERT-Score for evaluation and early stopping (optional).

    Args:
        train_data (List[Dict[str, str]]): Training data as a list of dictionaries with 'input' and 'output' keys.
        val_data (List[Dict[str, str]]): Validation data as a list of dictionaries with 'input' and 'output' keys.
        model_name (str, optional): Name or path of the pre-trained BLOOM model. Defaults to "bigscience/bloom-1b1".
        output_dir (str, optional): Directory to save the trained model. Defaults to "./models/bloom".
        num_train_epochs (int, optional): Number of training epochs. Defaults to 3.
        per_device_train_batch_size (int, optional): Training batch size per device (GPU/CPU). Defaults to 4.
        per_device_eval_batch_size (int, optional): Evaluation batch size per device (GPU/CPU). Defaults to 4.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 5e-5.
        gradient_accumulation_steps (int, optional): Number of gradient accumulation steps. Defaults to 1.
        early_stopping_patience (int, optional): Number of evaluation epochs with no improvement after which training will be stopped. Defaults to 2.

    Raises:
        SystemExit: Exits the program if training fails.
    """
    global tokenizer  # if you need to reuse in compute_metrics

    logger.info("Initializing tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # BLOOM often doesn't have a pad token by default; set to eos if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model moved to device: {device}")
    if device.type == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU found. Training will be on CPU, which may be slow.")

    logger.info("Preparing training dataset")
    train_dataset = prepare_dataset(train_data, tokenizer, max_input_length=1024, max_output_length=150)

    logger.info("Preparing validation dataset")
    val_dataset = prepare_dataset(val_data, tokenizer, max_input_length=1024, max_output_length=150)

    # For causal language modeling, we usually use DataCollatorForLanguageModeling with mlm=False
    logger.info("Setting up data collator")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Bloom is a causal LM, so no masked LM objective
    )

    logger.info("Defining training arguments")
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",     # Evaluate every epoch
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bertscore_f1",  # If you're using BERT-Score
        greater_is_better=True,
        logging_dir="./logs/bloom",
        logging_steps=50,
        save_strategy="epoch",
        fp16=True if device.type == "cuda" else False,
        dataloader_num_workers=2,
    )

    logger.info("Initializing Trainer with BERT-Score metric (compute_metrics)")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,  # Might be deprecated in future, see HF warnings
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer, remove_prompt_portion=True, prompt_delimiter="Summary:\n"),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    )

    logger.info("Starting BLOOM training")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        sys.exit(1)

    logger.info("Saving the trained BLOOM model and tokenizer")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Evaluating the trained model")
    try:
        metrics = trainer.evaluate()
        metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Evaluation metrics saved to {metrics_path}")
    except Exception as e:
        logger.error(f"Failed to evaluate the model: {e}")

    logger.info("BLOOM training completed successfully.")


def main():
    """
    Main entry point for training a BLOOM model with BERT-Score.
    It loads JSONL data, trains the model, and saves the result.

    This function performs the following steps:
        1. Loads training and validation data from JSONL files.
        2. Initiates the training process with specified hyperparameters.
        3. Saves the trained model and tokenizer.
    """
    data_dir = "data"
    train_file = os.path.join(data_dir, "train.jsonl")
    val_file = os.path.join(data_dir, "val.jsonl")

    logger.info("Loading training data")
    train_data = load_jsonl(train_file)
    logger.info(f"Loaded {len(train_data)} training samples.")

    logger.info("Loading validation data")
    val_data = load_jsonl(val_file)
    logger.info(f"Loaded {len(val_data)} validation samples.")

    train_bloom_model(
        train_data=train_data,
        val_data=val_data,
        model_name="bigscience/bloom-1b1",
        output_dir="./models/bloom",
        num_train_epochs=100,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        early_stopping_patience=3
    )


if __name__ == "__main__":
    main()
