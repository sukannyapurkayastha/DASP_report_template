# compute_metrics.py

import logging
import numpy as np
import torch
import evaluate

# Initialize logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    # Prevent adding multiple handlers in interactive environments
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.CRITICAL)

# Load BERT-Score metric
bertscore_metric = evaluate.load("bertscore")

def compute_metrics(
    eval_pred,
    tokenizer,
    remove_prompt_portion: bool = False,
    prompt_delimiter: str = None,
):
    """
    Compute BERT-Score between predictions and labels.
    Includes detailed debug logs to trace data shapes and types.

    Args:
        eval_pred (Tuple): The (predictions, labels) tuple from Trainer.
        tokenizer (PreTrainedTokenizer): The tokenizer used for decoding.
        remove_prompt_portion (bool): If True, tries to remove the prompt portion
            in each decoded string before scoring. Defaults to False.
        prompt_delimiter (str): If provided, is used to split out the prompt portion.
            For example, if your text is "PROMPT\\nOUTPUT", then prompt_delimiter="\\n"
            will remove everything up to the first newline.

    Returns:
        dict: A dictionary containing "bertscore_precision", "bertscore_recall",
              and "bertscore_f1".
    """
    logger.debug("[compute_metrics] Entering compute_metrics function.")

    predictions, labels = eval_pred

    # Debug: Check types
    logger.debug(f"[compute_metrics] Type of predictions: {type(predictions)}")
    logger.debug(f"[compute_metrics] Type of labels: {type(labels)}")

    # Handle predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
        logger.debug("[compute_metrics] Predictions were a tuple; took first element.")

    if isinstance(predictions, torch.Tensor):
        logger.debug(f"[compute_metrics] Predictions is a torch.Tensor with shape {predictions.shape}")
        if predictions.ndim == 3:
            logger.debug("[compute_metrics] Predictions have logits; applying argmax over last dimension.")
            predictions = predictions.argmax(dim=-1).cpu().numpy()
        else:
            predictions = predictions.cpu().numpy()
    elif isinstance(predictions, np.ndarray):
        logger.debug(f"[compute_metrics] Predictions is a numpy.ndarray with shape {predictions.shape}")
        if predictions.ndim == 3:
            logger.debug("[compute_metrics] Predictions have logits; applying argmax over last dimension.")
            predictions = np.argmax(predictions, axis=-1)
    elif isinstance(predictions, list):
        logger.debug("[compute_metrics] Predictions is a list; converting to numpy array.")
        predictions = np.array(predictions)
    else:
        logger.error(f"[compute_metrics] Unexpected type for predictions: {type(predictions)}")
        raise ValueError(f"Unexpected type for predictions: {type(predictions)}")

    # Handle labels
    if isinstance(labels, torch.Tensor):
        logger.debug(f"[compute_metrics] Labels is a torch.Tensor with shape {labels.shape}")
        labels = labels.cpu().numpy()
    elif isinstance(labels, np.ndarray):
        logger.debug(f"[compute_metrics] Labels is a numpy.ndarray with shape {labels.shape}")
    elif isinstance(labels, list):
        logger.debug("[compute_metrics] Labels is a list; converting to numpy array.")
        labels = np.array(labels)
    else:
        logger.error(f"[compute_metrics] Unexpected type for labels: {type(labels)}")
        raise ValueError(f"Unexpected type for labels: {type(labels)}")

    # Replace -100 with pad_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    logger.debug(f"[compute_metrics] Using pad_token_id={pad_id} to replace -100 in labels.")
    labels = np.where(labels == -100, pad_id, labels)

    # Ensure predictions and labels are 2D arrays
    if predictions.ndim != 2:
        logger.error(f"[compute_metrics] Predictions should be a 2D array, but got shape {predictions.shape}")
        raise ValueError(f"Predictions should be a 2D array, but got shape {predictions.shape}")
    if labels.ndim != 2:
        logger.error(f"[compute_metrics] Labels should be a 2D array, but got shape {labels.shape}")
        raise ValueError(f"Labels should be a 2D array, but got shape {labels.shape}")

    # Decode predictions and labels
    try:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        logger.debug("[compute_metrics] Successfully decoded predictions and labels.")
    except Exception as e:
        logger.error(f"[compute_metrics] Error during batch_decode: {e}")
        raise e

    # Optionally remove prompt portion from decoded strings
    if remove_prompt_portion and prompt_delimiter:
        logger.debug("[compute_metrics] Removing prompt portion using the given delimiter.")
        for i in range(len(decoded_preds)):
            if prompt_delimiter in decoded_preds[i]:
                # Split at the first occurrence of the delimiter, keep only the 'output' portion
                parts = decoded_preds[i].split(prompt_delimiter, 1)
                if len(parts) > 1:
                    decoded_preds[i] = parts[1]  # Keep text after the delimiter
            if prompt_delimiter in decoded_labels[i]:
                parts = decoded_labels[i].split(prompt_delimiter, 1)
                if len(parts) > 1:
                    decoded_labels[i] = parts[1]

    # Debug sample decoded strings
    if len(decoded_preds) > 0 and len(decoded_labels) > 0:
        logger.debug(f"[compute_metrics] Sample prediction: '{decoded_preds[0]}...'")
        logger.debug(f"[compute_metrics] Sample label: '{decoded_labels[0][:60]}...'")

    # Compute BERT-Score
    try:
        logger.debug("[compute_metrics] Computing BERT-Score.")
        bert_results = bertscore_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            model_type="bert-base-uncased"
        )
        logger.debug("[compute_metrics] BERT-Score computation successful.")
    except Exception as e:
        logger.error(f"[compute_metrics] Error during BERT-Score computation: {e}")
        raise e

    # Aggregate scores
    precision_arr = bert_results["precision"]
    recall_arr = bert_results["recall"]
    f1_arr = bert_results["f1"]

    precision = np.mean(precision_arr) * 100
    recall = np.mean(recall_arr) * 100
    f1 = np.mean(f1_arr) * 100

    metrics = {
        "bertscore_precision": round(precision, 2),
        "bertscore_recall": round(recall, 2),
        "bertscore_f1": round(f1, 2)
    }

    logger.debug(f"[compute_metrics] BERT-Score Metrics: {metrics}")
    return metrics
