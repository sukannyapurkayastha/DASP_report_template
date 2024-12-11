import torch
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForSequenceClassification
from loguru import logger

# Import helper functions from the fine-grained request classifier module
from classification.fine_request_classifier_llama import (
    map_prediction_to_label,
    generate_predictions_from_dataset,
    few_shot_examples,
    label_map,
)


def process_dataframe_request(df: pd.DataFrame, local_dir: str, local_dir_fine_request: str) -> pd.DataFrame:
    """
    Processes a DataFrame containing sentences for coarse and fine-grained classification.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a column `sentence` containing text to classify.
    local_dir (str): Local directory of the binary classifier.
    local_dir_fine_request (str): Local directory of the multiclass classifier.

    Returns:
    pd.DataFrame: DataFrame with added columns for coarse and fine-grained labels.
    """
    if 'sentence' not in df.columns:
        raise ValueError("DataFrame must include a 'sentence' column.")

    texts = df['sentence'].tolist()

    # Load the coarse-grained request classifier
    tokenizer_bert = BertTokenizer.from_pretrained(local_dir)
    model_bert = BertForSequenceClassification.from_pretrained(local_dir)
    model_bert.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_bert.to(device)

    # Define a prediction function for the coarse-grained classifier
    def predict_request(texts):
        """
        Predicts coarse-grained labels for a batch of texts.

        Parameters:
        texts (list): List of input sentences.

        Returns:
        np.ndarray: Array of predicted labels.
        """
        inputs = tokenizer_bert(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model_bert(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        return predictions.cpu().numpy()

    # Perform coarse-grained classification
    logger.info("Classifying requests (coarse-grained)")
    predictions_bert = []
    batch_size = 16
    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            preds = predict_request(batch_texts)
            predictions_bert.extend(preds)
    except Exception as e:
        print(f"Error during prediction: {e}")

    df['coarse_label_pred'] = predictions_bert

    # Filter rows where coarse-grained prediction indicates a request
    df_requests = df[df['coarse_label_pred'] == 1].copy()

    # Load the fine-grained request classifier
    logger.info("Loading fine-grained T5 model")
    model_path_fine_request_classifier = "backend/models/request_classifier/fine_request_classifier/"
    tokenizer_t5 = T5Tokenizer.from_pretrained(local_dir_fine_request)
    model_t5 = T5ForConditionalGeneration.from_pretrained(local_dir_fine_request)
    model_t5.eval()
    device_t5 = "cuda" if torch.cuda.is_available() else "cpu"
    model_t5.to(device_t5)
    tokenizer_t5.pad_token_id = tokenizer_t5.eos_token_id

    # Perform fine-grained classification
    if not df_requests.empty:
        logger.info("Performing fine-grained classification")
        predictions_t5 = generate_predictions_from_dataset(
            df_requests[['sentence']], few_shot_examples, tokenizer_t5, model_t5
        )

        # Map predictions to fine-grained labels
        mapped_predictions = [map_prediction_to_label(pred, label_map) for pred in predictions_t5]
        df_requests['fine_grained_label'] = mapped_predictions

        # Map numeric labels to their corresponding category names
        reverse_label_map = {v: k for k, v in label_map.items()}
        df_requests['fine_grained_label_name'] = df_requests['fine_grained_label'].map(reverse_label_map)
    else:
        # If no requests are detected, set fine-grained labels to default values
        df_requests['fine_grained_label'] = -1
        df_requests['fine_grained_label_name'] = None

    return df_requests
