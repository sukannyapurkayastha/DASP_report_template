# predict_compare.py

import os
import json
import logging
from typing import List, Dict

from data.load_data import load_jsonl
from compute_metrics import compute_metrics

# Import predict functions from each model's prediction script
from predict_T5 import predict as predict_T5, load_T5_model
from predict_BART import predict as predict_BART, load_BART_model
from predict_BLOOM import predict as predict_BLOOM, load_BLOOM_model
from predict_LLAMA2 import predict as predict_LLAMA2, load_LLAMA2_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to load test data, generate predictions using each model,
    compute evaluation metrics, and compare the performance of each model.
    """
    test_file = "data/test.jsonl"
    logger.info(f"Loading test data from {test_file}")
    test_data = load_jsonl(test_file)

    if not test_data:
        logger.error("No test data found. Exiting.")
        return

    # Extract inputs and labels
    inputs = [entry['input'] for entry in test_data]
    labels = [entry['output'] for entry in test_data]

    # Initialize a dictionary to store predictions for each model
    model_predictions: Dict[str, List[str]] = {
        "T5": [],
        "BART": [],
        "BLOOM": [],
        "LLaMA2": []
    }

    # Load tokenizers and each model
    model_T5, tokenizer_T5 = load_T5_model(model_dir = "./models/t5")
    model_BART, tokenizer_BART = load_BART_model(model_dir = "./models/bart")
    model_BLOOM, tokenizer_BLOOM = load_BLOOM_model(model_dir = "./models/bloom")
    model_LLAMA2, tokenizer_LLAMA2 = load_LLAMA2_model()

    # Generate predictions for each model
    for idx, input_text in enumerate(inputs):
        logger.info(f"Processing sample {idx+1}/{len(inputs)}")
        try:
            pred_t5 = predict_T5(input_text=input_text, model=model_T5, tokenizer=tokenizer_T5)
            model_predictions["T5"].append(pred_t5)
        except Exception as e:
            logger.error(f"T5 prediction failed for sample {idx+1}: {e}")
            model_predictions["T5"].append("")

        try:
            pred_bart = predict_BART(input_text=input_text, model=model_BART, tokenizer=tokenizer_BART)
            model_predictions["BART"].append(pred_bart)
        except Exception as e:
            logger.error(f"BART prediction failed for sample {idx+1}: {e}")
            model_predictions["BART"].append("")

        try:
            pred_bloom = predict_BLOOM(input_text=input_text, model=model_BLOOM, tokenizer=tokenizer_BLOOM)
            model_predictions["BLOOM"].append(pred_bloom)
        except Exception as e:
            logger.error(f"BLOOM prediction failed for sample {idx+1}: {e}")
            model_predictions["BLOOM"].append("")

        try:
            pred_llama2 = predict_LLAMA2(input_text=input_text, model=model_LLAMA2, tokenizer=tokenizer_LLAMA2)
            model_predictions["LLaMA2"].append(pred_llama2)
        except Exception as e:
            logger.error(f"LLaMA2 prediction failed for sample {idx+1}: {e}")
            model_predictions["LLaMA2"].append("")

    # Encode predictions and labels for compute_metrics
    model_metrics: Dict[str, Dict[str, float]] = {}

    for model_name, predictions in model_predictions.items():
        logger.info(f"Computing metrics for model: {model_name}")
        try:
            if model_name == "T5":
                tokenizer = tokenizer_T5
            elif model_name == "BART":
                tokenizer = tokenizer_BART
            elif model_name == "BLOOM":
                tokenizer = tokenizer_BLOOM
            elif model_name == "LLaMA2":
                tokenizer = tokenizer_LLAMA2
            else:
                logger.warning(f"No tokenizer found for model: {model_name}")
                tokenizer = None

            if tokenizer is None:
                raise ValueError(f"Tokenizer for model {model_name} is not loaded.")

            # Encode predictions and labels
            # Replace empty predictions with tokenizer.pad_token or appropriate token
            encoded_predictions = [
                pred if pred else tokenizer.pad_token for pred in predictions
            ]
            encoded_labels = [
                label if label else tokenizer.pad_token for label in labels
            ]

            # Tokenize the predictions and labels to get token IDs
            tokenized_predictions = tokenizer(
                encoded_predictions,
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="np"
            )["input_ids"]

            tokenized_labels = tokenizer(
                encoded_labels,
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="np"
            )["input_ids"]

            # Compute metrics
            metrics = compute_metrics((tokenized_predictions, tokenized_labels), tokenizer=tokenizer)
            model_metrics[model_name] = metrics
            logger.info(f"Metrics for {model_name}: {metrics}")
        except Exception as e:
            logger.error(f"Error computing metrics for model {model_name}: {e}")
            model_metrics[model_name] = {}

    # Display the comparison of metrics
    print("\n=== Model Performance Comparison ===\n")
    for model_name, metrics in model_metrics.items():
        print(f"Model: {model_name}")
        if metrics:
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value}")
        else:
            print("  Metrics not available due to errors.")
        print()

    # Save the comparison to a JSON file
    comparison_file = "model_performance_comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(model_metrics, f, indent=4)
    logger.info(f"Model performance comparison saved to {comparison_file}")


if __name__ == "__main__":
    main()
