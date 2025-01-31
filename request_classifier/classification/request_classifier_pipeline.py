import torch
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForSequenceClassification
from loguru import logger

# Set higher timeout to prevent ReadTimeout
# https://github.com/huggingface/transformers/issues/12575
import socket
from urllib3.connection import HTTPConnection

HTTPConnection.default_socket_options = (
        HTTPConnection.default_socket_options + [
    (socket.SOL_SOCKET, socket.SO_SNDBUF, 2000000),
    (socket.SOL_SOCKET, socket.SO_RCVBUF, 2000000)
])

label_map = {
    "Request for Improvement": 0,
    "Request for Explanation": 1,
    "Request for Experiment": 2,
    "Request for Typo Fix": 3,
    "Request for Clarification": 4,
    "Request for Result": 5,
    "Request unclear": -1,
}

fine_to_category_map = {
    "arg-request_edit": "Request for Improvement",
    "arg-request_explanation": "Request for Explanation",
    "arg-request_experiment": "Request for Experiment",
    "arg-request_typo": "Request for Typo Fix",
    "arg-request_clarification": "Request for Clarification",
    "arg-request_result": "Request for Result",
}


def summarize_requests_by_authors(df_requests: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates fine-grained labels and summarizes them by the number of unique authors 
    and associated comments, with sentences grouped by their respective categories.

    Parameters:
    df_requests (pd.DataFrame): DataFrame containing fine-grained request information.

    Returns:
    pd.DataFrame: Summary DataFrame with categories, unique author counts, 
                  associated comments, and sentences grouped by their respective labels.
    """
    # Count total number of unique authors
    num_authors = df_requests['author'].nunique()

    # Create a summary grouped by fine-grained label names
    summary = (
        df_requests.groupby('fine_grained_label_name')
        .agg(
            Frequency=('author', lambda x: x.nunique() / num_authors),  # Relative frequency of unique authors
        )
        .reset_index()
        .rename(columns={'fine_grained_label_name': 'Request Information'})  # Rename columns for clarity
        .sort_values(by='Frequency', ascending=False)  # Sort by proportion
    )

    # Add sentences grouped by authors for each fine-grained label
    def group_sentences_by_label(df: pd.DataFrame) -> dict:
        """
        Groups sentences by authors for each fine-grained label.

        Parameters:
        df (pd.DataFrame): DataFrame containing fine-grained labels, authors, and sentences.

        Returns:
        dict: A dictionary where keys are fine-grained labels, and values are lists of 
              [author, [list of sentences written by the author]] for that label.
        """
        grouped = {}
        for label in df['fine_grained_label_name'].unique():
            label_df = df[df['fine_grained_label_name'] == label]
            author_sentences = label_df.groupby('author')['sentence'].apply(list).reset_index()
            grouped[label] = author_sentences.apply(lambda row: [row['author'], row['sentence']], axis=1).tolist()
        return grouped

    # Generate sentences grouped by authors for each fine-grained label
    sentences_by_label = group_sentences_by_label(df_requests)

    # Add grouped sentences for each category to the summary
    summary['Comments'] = summary['Request Information'].map(sentences_by_label)
    summary['Request Information'] = summary['Request Information'].str.replace("Request for ", "", regex=False)

    return summary


def process_dataframe_request(df: pd.DataFrame, local_dir_request: str, hf_request_classifier: str,
                              local_dir_fine_request: str, hf_fine_request_classifier: str) -> pd.DataFrame:
    """
    Processes a DataFrame containing sentences for coarse and fine-grained classification.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a column `sentence` containing text to classify.
    local_dir_request (str): Local directory of the binary classifier.
    hf_request_classifier (str): Hugging Face model path for the binary request classifier.
    local_dir_fine_request (str): Local directory of the multiclass classifier.
    hf_fine_request_classifier (str): Hugging Face model path for the multiclass classifier.

    Returns:
    pd.DataFrame: DataFrame with added columns for coarse and fine-grained labels.
    """
    if 'sentence' not in df.columns:
        raise ValueError("DataFrame must include a 'sentence' column.")

    texts = df['sentence'].tolist()

    # Load the coarse-grained request classifier
    tokenizer_bert = BertTokenizer.from_pretrained(hf_request_classifier, cache_dir=local_dir_request)
    model_bert = BertForSequenceClassification.from_pretrained(hf_request_classifier, cache_dir=local_dir_request)
    # tokenizer_bert = load_BertTokenizer(local_path=local_dir, huggingface_model_path=hf_request_classifier)
    # model_bert = load_BertForSequenceClassification(local_path=local_dir, huggingface_model_path=hf_request_classifier)
    model_bert.eval()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
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
    # print(df_requests)

    logger.info("Loading fine-grained BERT model")
    tokenizer_bert_fine = BertTokenizer.from_pretrained(hf_fine_request_classifier, cache_dir=local_dir_fine_request)
    model_bert_fine = BertForSequenceClassification.from_pretrained(hf_fine_request_classifier,
                                                                    cache_dir=local_dir_fine_request)
    # tokenizer_bert_fine = load_BertTokenizer(local_path=local_dir_fine_request,
    #                                          huggingface_model_path=hf_fine_request_classifier)
    # model_bert_fine = load_BertForSequenceClassification(local_path=local_dir_fine_request,
    #                                                      huggingface_model_path=hf_fine_request_classifier)
    # tokenizer_bert_fine = BertTokenizer.from_pretrained(local_dir_fine_request)
    # model_bert_fine = BertForSequenceClassification.from_pretrained(local_dir_fine_request)
    model_bert_fine.eval()
    device_fine = "cuda" if torch.cuda.is_available() else "cpu"
    model_bert_fine.to(device_fine)

    def predict_request_fine(batch_texts):
        """
        Predicts fine-grained (multi-class) labels for a batch of texts.
        Returns an array of numeric label IDs.
        """
        inputs = tokenizer_bert_fine(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(device_fine) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model_bert_fine(**inputs)
        logits = outputs.logits
        return torch.argmax(logits, dim=-1).cpu().numpy()

    # ------------------------------------------------------------------#
    # 4) Perform fine-grained classification (only for request rows)
    # ------------------------------------------------------------------#
    if not df_requests.empty:
        logger.info("Performing fine-grained classification with second BERT")
        texts_fine = df_requests['sentence'].tolist()
        predictions_fine = []
        for i in tqdm(range(0, len(texts_fine), batch_size), desc="Processing fine batches"):
            batch_texts_fine = texts_fine[i:i + batch_size]
            preds = predict_request_fine(batch_texts_fine)
            predictions_fine.extend(preds)

        # Assign numeric label IDs
        df_requests['fine_grained_label'] = predictions_fine

        # Convert numeric label IDs to label names
        # Example: label_map = {"Request for Explanation": 0, "Request for Improvement": 1, ...}
        # So we invert that mapping to {0: "Request for Explanation", 1: ...}
        reverse_label_map = {v: k for k, v in label_map.items()}
        df_requests['fine_grained_label_name'] = df_requests['fine_grained_label'].map(reverse_label_map)

    else:
        # If no requests are detected, set fine-grained labels to default values
        df_requests['fine_grained_label'] = -1
        df_requests['fine_grained_label_name'] = None

    # ------------------------------------------------------------------#
    # 5) Summarize results
    # ------------------------------------------------------------------#
    df_requests_summarized = summarize_requests_by_authors(df_requests)
    # print(df_requests_summarized)
    return df_requests_summarized
