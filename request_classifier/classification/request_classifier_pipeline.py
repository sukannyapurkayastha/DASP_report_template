import torch
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForSequenceClassification
import sys
import os
from huggingface_hub import hf_hub_download, list_repo_files

# module_path = os.path.abspath("model_training/nlp/request_classifier/models/classification")
# sys.path.append(module_path)


# from model_training.nlp.request_classifier.models.classification.request_classifier import predict as predict_request
from models.request_classifier.request_classifier.fine_request_classifier_llama import (
    # create_few_shot_prompt,
    map_prediction_to_label,
    generate_predictions_from_dataset,
    few_shot_examples,
    label_map,
)

def process_dataframe_request(df: pd.DataFrame) -> pd.DataFrame:
    
    if 'sentence' not in df.columns:
        raise ValueError("Dataframe has to include sentences.")
    
    texts = df['sentence'].tolist()

    repo_id = "JohannesLemken/DASP_models"

    # List all files in the repository
    all_files = list_repo_files(repo_id=repo_id, repo_type="model")

    # Filter the files to only those in the "Request_Classifier/RequestClassifier" directory
    subdir = "Request_Classifier/RequestClassifier/"
    files_to_download = [f for f in all_files if f.startswith(subdir)]

    # Create a local directory to store these files, if desired
    local_dir = "backend/models/request_classifier/request_classifier/"
    os.makedirs(local_dir, exist_ok=True)

    # Download each file
    for filename in files_to_download:
        local_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision="main"  # or a specific branch/tag/commit if needed
        )

        # Optionally move it into our local directory to preserve structure
        # Extract the relative file path after `subdir`
        relative_path = filename[len(subdir):]
        local_subpath = os.path.join(local_dir, relative_path)

        # Create any necessary nested directories
        os.makedirs(os.path.dirname(local_subpath), exist_ok=True)

        # Move/rename the downloaded file into our target structure
        os.replace(local_file_path, local_subpath)

        print(f"Downloaded and saved {filename} to {local_subpath}")

    model_path_request_classifier = "backend/models/request_classifier/request_classifier/"
    tokenizer_bert = BertTokenizer.from_pretrained(local_dir)
    model_bert = BertForSequenceClassification.from_pretrained(local_dir)
    model_bert.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_bert.to(device)

    def predict_request(texts):
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
    
    print("Classifying requests...")
    predictions_bert = []
    batch_size = 16
    try:
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            preds = predict_request(batch_texts)
            predictions_bert.extend(preds)
    except Exception as e:
        print(e)

    df['coarse_label_pred'] = predictions_bert

    
    df_requests = df[df['coarse_label_pred'] == 1].copy()

    print("Loading T5 model...")
    model_path_fine_request_classifier = "backend/models/request_classifier/fine_request_classifier/" 
    tokenizer_t5 = T5Tokenizer.from_pretrained(model_path_fine_request_classifier)
    model_t5 = T5ForConditionalGeneration.from_pretrained(model_path_fine_request_classifier)
    model_t5.eval()
    device_t5 = "cuda" if torch.cuda.is_available() else "cpu"
    model_t5.to(device_t5)
    tokenizer_t5.pad_token_id = tokenizer_t5.eos_token_id

    if not df_requests.empty:
        print("Fine-grained classification...")
        predictions_t5 = generate_predictions_from_dataset(
            df_requests[['sentence']], few_shot_examples, tokenizer_t5, model_t5
        )

        mapped_predictions = [map_prediction_to_label(pred, label_map) for pred in predictions_t5]
        
        df_requests['fine_grained_label'] = mapped_predictions
        reverse_label_map = {v: k for k, v in label_map.items()}
        df_requests['fine_grained_label_name'] = df_requests['fine_grained_label'].map(reverse_label_map)
    else:
        df_requests['fine_grained_label'] = -1
        df_requests['fine_grained_label_name'] = None

    return df_requests

if __name__ == "__main__":
    csv_file_path = "C:/Users/Johannes/Downloads/test.csv"
    df_input = pd.read_csv(csv_file_path)
    df_input = df_input.dropna(subset=['sentence'])
    df_result = process_dataframe_request(df_input)
    print(df_result)
