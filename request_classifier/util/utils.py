import os
from loguru import logger
from transformers import BertTokenizer, BertForSequenceClassification


def load_BertTokenizer(local_path: str, huggingface_model_path: str):
    """
    Load the load_BertTokenizer, checking for local availability.
    :param local_path: Local path of the BertTokenizer.
    :param huggingface_model_path: Hugging Face model path.
    :return: load_BertTokenizer
    """
    if os.path.exists(local_path) and os.path.isdir(local_path):
        # Check if required tokenizer files exist in the directory
        tokenizer_files = ["vocab.txt", "tokenizer.json", "config.json"]
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


def load_BertForSequenceClassification(local_path: str, huggingface_model_path: str):
    """
    Load the BertForSequenceClassification, checking for local availability.
        :param local_path: Local path of the BertTokenizer.
    :param huggingface_model_path: Hugging Face model path.
    :return: BertForSequenceClassification
    """
    if os.path.exists(local_path) and os.path.isdir(local_path):
        # Check if required model files exist in the directory
        model_files = ["config.json", "model.safetensor"]
        if all(os.path.exists(os.path.join(local_path, f)) for f in model_files):
            logger.info(f"Loading model from local path: {local_path}")
            model = BertForSequenceClassification.from_pretrained(local_path)
            return model
        else:
            logger.info("Model files are missing locally.")
    else:
        logger.info("Local path does not exist or is not a directory.")

    # If the local model is not available, download from Hugging Face
    logger.info(f"Downloading model from Hugging Face: {huggingface_model_path}")
    model = BertForSequenceClassification.from_pretrained(huggingface_model_path)
    return model
