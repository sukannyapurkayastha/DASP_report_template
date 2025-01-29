import os
from loguru import logger

from transformers import (
    DistilBertTokenizer,
    TFDistilBertForSequenceClassification,
    BertForSequenceClassification,
    BertTokenizer
)


def load_DistilBertTokenizer():
    """
    Load the DistilBertTokenizer, checking for local availability.
    :return: DistilBertTokenizer
    """

    local_path = "models/attitude_root/"
    huggingface_model_path = "DASP-ROG/AttitudeModel"

    if os.path.exists(local_path) and os.path.isdir(local_path):
        # Check if required tokenizer files exist in the directory
        tokenizer_files = ["vocab.txt", "tokenizer_config.json", "config.json"]
        if all(os.path.exists(os.path.join(local_path, f)) for f in tokenizer_files):
            logger.info(f"Loading tokenizer from local path: {local_path}")
            tokenizer = DistilBertTokenizer.from_pretrained(local_path)
            return tokenizer
        else:
            logger.info("Tokenizer files are missing locally.")
    else:
        logger.info("Local path does not exist or is not a directory.")

    # If the local tokenizer is not available, download from Hugging Face
    logger.info(f"Downloading tokenizer from Hugging Face: {huggingface_model_path}")
    tokenizer = DistilBertTokenizer.from_pretrained(huggingface_model_path)
    return tokenizer


def load_TFDistilBertForSequenceClassification(num_labels: int = 9):
    """
    Load the TFDistilBertForSequenceClassification, checking for local availability.
    :return: TFDistilBertForSequenceClassification
    """

    local_path = "models/attitude_root/"
    huggingface_model_path = "DASP-ROG/AttitudeModel"

    if os.path.exists(local_path) and os.path.isdir(local_path):
        # Check if required model files exist in the directory
        model_files = ["config.json", "tf_model.h5"]
        if all(os.path.exists(os.path.join(local_path, f)) for f in model_files):
            logger.info(f"Loading model from local path: {local_path}")
            model = TFDistilBertForSequenceClassification.from_pretrained(local_path, num_labels=num_labels)
            return model
        else:
            logger.info("Model files are missing locally.")
    else:
        logger.info("Local path does not exist or is not a directory.")

    # If the local model is not available, download from Hugging Face
    logger.info(f"Downloading model from Hugging Face: {huggingface_model_path}")
    model = TFDistilBertForSequenceClassification.from_pretrained(local_path, num_labels=num_labels)
    return model


def load_BertTokenizer():
    """
    Load the load_BertTokenizer, checking for local availability.
    :return: load_BertTokenizer
    """

    local_path = "models/attitude_theme/"
    huggingface_model_path = "DASP-ROG/ThemeModel"

    if os.path.exists(local_path) and os.path.isdir(local_path):
        # Check if required tokenizer files exist in the directory
        tokenizer_files = ["vocab.txt", "tokenizer_config.json", "config.json"]
        if all(os.path.exists(os.path.join(local_path, f)) for f in tokenizer_files):
            logger.info(f"Loading tokenizer from local path: {local_path}")
            tokenizer = BertTokenizer.from_pretrained(local_path)
            return tokenizer
        else:
            logger.info("Tokenizer files are missing locally.")
    else:
        logger.info("Local path does not exist or is not a directory.")

    # If the local tokenizer is not available, download from Hugging Face
    logger.info(f"Downloading tokenizer from Hugging Face: {huggingface_model_path}")
    tokenizer = BertTokenizer.from_pretrained(huggingface_model_path)
    return tokenizer


def load_BertForSequenceClassification(num_labels: int = 11):
    """
    Load the BertForSequenceClassification, checking for local availability.
    :return: BertForSequenceClassification
    """

    local_path = "models/attitude_theme/"
    huggingface_model_path = "DASP-ROG/ThemeModel"

    if os.path.exists(local_path) and os.path.isdir(local_path):
        # Check if required model files exist in the directory
        model_files = ["config.json", "model.safetensors"]
        if all(os.path.exists(os.path.join(local_path, f)) for f in model_files):
            logger.info(f"Loading model from local path: {local_path}")
            model = BertForSequenceClassification.from_pretrained(local_path, num_labels=num_labels,
                                                                  problem_type="multi_label_classification")
            return model
        else:
            logger.info("Model files are missing locally.")
    else:
        logger.info("Local path does not exist or is not a directory.")

    # If the local model is not available, download from Hugging Face
    logger.info(f"Downloading model from Hugging Face: {huggingface_model_path}")
    model = BertForSequenceClassification.from_pretrained(huggingface_model_path, num_labels=num_labels,
                                                          problem_type="multi_label_classification")
    return model
