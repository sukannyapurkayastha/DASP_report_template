"""
train_T5.py

This module provides functionality to train a T5 model (T5-large) 
with a BERT-Score metric, using the Hugging Face Trainer

"""

import os
import logging
from typing import List, Dict, Tuple

import torch
import numpy as np
from transformers import (
    T5TokenizerFast,                  # Use T5 tokenizer
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
    GenerationConfig,                  # If needed for T5
)
from datasets import Dataset
from data.load_data import load_jsonl  # Ensure this path is correct
import evaluate
import warnings

# Suppress specific warnings if necessary (optional)
warnings.filterwarnings(
    "ignore",
    message="`tokenizer` is deprecated and will be removed in version 5.0.0 for `Seq2SeqTrainer.__init__`. Use `processing_class` instead.",
    category=FutureWarning
)

warnings.filterwarnings(
    "ignore",
    message="`evaluation_strategy` is deprecated and will be removed in version 4.46 of Transformers. Use `eval_strategy` instead",
    category=FutureWarning
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    level=logging.CRITICAL,
)
logger = logging.getLogger(__name__)

# Global variables for tokenizer and model
tokenizer = None
model = None
bertscore = None

def preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
    """
    Tokenize the input and output texts.

    Args:
        examples (Dict[str, List[str]]): A dictionary with 'input' and 'output' keys containing lists of strings.

    Returns:
        Dict[str, List[int]]: A dictionary with tokenized inputs and labels.
    """
    # Optional: Add task-specific prefixes if necessary
    # For example, for summarization:
    # inputs = [f"summarize: {inp}" for inp in examples["input"]]
    inputs = examples["input"]  # Modify if you want to add prefixes
    targets = examples["output"]
    
    model_inputs = tokenizer(
        inputs,
        max_length=1024,
        padding="max_length",  # Explicitly set padding to 'max_length'
        truncation=True,       # Enable truncation
    )
    labels = tokenizer(
        targets,
        max_length=180,
        padding="max_length",  # Explicitly set padding to 'max_length'
        truncation=True,       # Enable truncation
        text_target=targets,   # Use text_target for T5
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred: Tuple[List[List[int]], List[List[int]]]) -> Dict[str, float]:
    """
    Compute BERTScore metrics for model predictions, handling -100 tokens appropriately.
    
    Args:
        eval_pred (Tuple[List[List[int]], List[List[int]]]): A tuple containing two lists:
            - predictions: List of predicted token IDs.
            - labels: List of true token IDs, with -100 indicating tokens to ignore.
    
    Returns:
        Dict[str, float]: A dictionary containing the BERTScore precision, recall, and F1 scores.
    """

    predictions, labels = eval_pred

    # Convert lists to NumPy arrays for efficient manipulation
    labels = np.array(labels)

    # Replace -100 with the tokenizer's pad_token_id to avoid decoding errors
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode the predictions and labels into strings
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Optional: Strip whitespace for cleaner evaluation
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Initialize the BERTScore metric
    bertscore = evaluate.load("bertscore")

    # Compute BERTScore
    bertscore_results = bertscore.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        lang="en",  # Adjust language if necessary
        verbose=False,
    )

    # Aggregate the BERTScore results
    metrics = {
        "bertscore_precision": np.mean(bertscore_results["precision"]),
        "bertscore_recall": np.mean(bertscore_results["recall"]),
        "bertscore_f1": np.mean(bertscore_results["f1"]),
    }

    return metrics

def main():
    """
    Main function to load data, initialize model and tokenizer, and start training.
    """
    global tokenizer, model, bertscore

    # 1. Load Data
    data_dir = "data"
    train_file = os.path.join(data_dir, "train.jsonl")
    val_file = os.path.join(data_dir, "val.jsonl")

    logger.info("Loading training data from %s", train_file)
    train_data = load_jsonl(train_file)
    logger.info("Loaded %d training examples", len(train_data))

    logger.info("Loading validation data from %s", val_file)
    val_data = load_jsonl(val_file)
    logger.info("Loaded %d validation examples", len(val_data))

    # 2. Convert List of Dicts to Dict of Lists for Hugging Face Dataset
    logger.info("Converting training data to Hugging Face Dataset format...")
    train_dict = {
        "input": [item["input"] for item in train_data],
        "output": [item["output"] for item in train_data],
    }
    train_dataset = Dataset.from_dict(train_dict)

    logger.info("Converting validation data to Hugging Face Dataset format...")
    val_dict = {
        "input": [item["input"] for item in val_data],
        "output": [item["output"] for item in val_data],
    }
    val_dataset = Dataset.from_dict(val_dict)

    # 3. Initialize Tokenizer and Model
    model_name = "t5-large"  # Use a T5 model variant
    tokenizer = T5TokenizerFast.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True,  # Handle missing keys gracefully
    )

    # Log and verify vocab sizes
    logger.info(f"Tokenizer class: {type(tokenizer)}")
    logger.info(f"Model class: {type(model)}")
    logger.info(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    logger.info(f"Model vocabulary size: {model.config.vocab_size}")
    #assert tokenizer.vocab_size == model.config.vocab_size, "Tokenizer and model vocab sizes do not match!"

    # Log model's special tokens
    logger.info(f"Model eos_token_id: {model.config.eos_token_id}")
    logger.info(f"Model pad_token_id: {model.config.pad_token_id}")
    logger.info(f"Model decoder_start_token_id: {model.config.decoder_start_token_id}")

    # **Set decoder_start_token_id to pad_token_id for T5**
    model.config.decoder_start_token_id = tokenizer.pad_token_id
    logger.info(f"Set model decoder_start_token_id to tokenizer.pad_token_id: {model.config.decoder_start_token_id}")

    # 4. Test Model Generation to Ensure It Doesn't Produce -100
    def test_model_generation():
        """
        Test the model's generation to ensure it does not produce invalid token IDs.
        """
        test_sentence = "This is a test sentence."
        inputs = tokenizer(
            test_sentence,
            return_tensors="pt",
            max_length=1024,
            padding="max_length",
            truncation=True,
        ).to(model.device)
        outputs = model.generate(**inputs)
        generated_ids = outputs[0].tolist()
        logger.info(f"Generated token IDs: {generated_ids}")
        if -100 in generated_ids:
            logger.error("Model generated -100 token ID, which is invalid.")
        else:
            logger.info("Model generation is valid.")
        
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Decoded output: {decoded_output}")

    test_model_generation()

    # 5. Tokenize the Datasets
    try:
        logger.info("Tokenizing the training dataset...")
        tokenized_train = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=["input", "output"],
        )
    except Exception as e:
        logger.error(f"Error during tokenizing training dataset: {e}")
        raise e

    try:
        logger.info("Tokenizing the validation dataset...")
        tokenized_val = val_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=["input", "output"],
        )
    except Exception as e:
        logger.error(f"Error during tokenizing validation dataset: {e}")
        raise e

    # 6. Load BERTScore Metric
    bertscore = evaluate.load("bertscore")

    # 7. Define Training Arguments
    output_dir = "./models/t5"
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=100,  # Adjust as needed
        per_device_train_batch_size=4,  # Adjust based on GPU memory
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="epoch",            # Updated from evaluation_strategy
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="bertscore_f1",
        greater_is_better=True,
        fp16=True,                        # Enable mixed precision if supported
        optim="adamw_torch",
        predict_with_generate=True,
        generation_max_length=180,        # Address max_length warning
    )

    # 8. Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding="max_length",  # Explicitly set padding to 'max_length'
        max_length=1024,
        label_pad_token_id=-100,
    )

    # 9. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,  # Correct parameter
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Stop after 3 epochs with no improvement
    )

    # 10. Utilize CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info("Training on %s", device)

    # 11. Start Training
    logger.info("Starting training...")
    trainer.train()

    # 12. Save the final model
    logger.info("Saving the model to %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training complete!")

if __name__ == "__main__":
    """
    Entry point of the training script. Tests the tokenizer and model with a sample sentence before initiating training.
    """
    # Initialize variables for testing
    test_sentence = "This is a test sentence."
    model_name = "t5-large"  # Use the same model as in the main function

    # Load tokenizer and model for testing
    try:
        tokenizer = T5TokenizerFast.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True,  # Handle missing keys gracefully
        )
        inputs = tokenizer(
            test_sentence,
            return_tensors="pt",
            max_length=1024,
            padding="max_length",
            truncation=True,
        ).to(model.device)
        outputs = model.generate(**inputs)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test decoded output: {decoded}")
    except Exception as e:
        logger.error(f"Error during tokenizer-model test: {e}")
        raise e

    # Proceed to main training function
    main()
