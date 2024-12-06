import torch
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForSequenceClassification
import sys
import os



module_path = os.path.abspath("model_training/nlp/request_classifier/models/classification")
sys.path.append(module_path)


from request_classifier import predict as predict_request
from fine_request_classifier_llama import (
    create_few_shot_prompt,
    map_prediction_to_label,
    generate_predictions_from_dataset,
    few_shot_examples,
    label_map,
)

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    
    if 'sentence' not in df.columns:
        raise ValueError("Dataframe has to include sentences.")
    
    texts = df['sentence'].tolist()

   
    model_path = "./bert_request_classifier_model"  
    tokenizer_bert = BertTokenizer.from_pretrained(model_path)
    model_bert = BertForSequenceClassification.from_pretrained(model_path)
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
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        preds = predict_request(batch_texts)  
        predictions_bert.extend(preds)

    df['coarse_label_pred'] = predictions_bert

    
    df_requests = df[df['coarse_label_pred'] == 1].copy()

    print("Loading T5 model...")
    tokenizer_t5 = T5Tokenizer.from_pretrained("google/flan-t5-xl", legacy=False)
    model_t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
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
    df_result = process_dataframe(df_input)
    print(df_result)
