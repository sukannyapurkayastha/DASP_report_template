
import torch
import pandas as pd
from tqdm import tqdm
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


def main():

    print(module_path)
 
    test_file = "backend/nlp/request_classifier/DISAPERE/final_dataset/Request/test.csv"
    test_data = pd.read_csv(test_file)
    texts = test_data['text'].tolist()

   
    print("Classifying with BERT model...")
    predictions_bert = []
    batch_size = 16
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        preds = predict_request(batch_texts)
        predictions_bert.extend(preds)

   
    request_indices = [i for i, pred in enumerate(predictions_bert) if pred == 1]
    request_sentences = [texts[i] for i in request_indices]

    
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    tokenizer_t5 = T5Tokenizer.from_pretrained("google/flan-t5-xl", legacy=False)
    model_t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
    model_t5.eval()
    model_t5.to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_t5.pad_token_id = tokenizer_t5.eos_token_id

    
    request_data = pd.DataFrame({'text': request_sentences})

    
    print("Classifying with T5 model...")
    predictions_t5 = generate_predictions_from_dataset(
        request_data, few_shot_examples, tokenizer_t5, model_t5
    )


    mapped_predictions = [map_prediction_to_label(pred, label_map) for pred in predictions_t5]

   
    test_data['coarse_label_pred'] = predictions_bert 
    test_data['fine_grained_label'] = -1  

    for idx_in_list, idx_in_df in enumerate(request_indices):
        test_data.at[idx_in_df, 'fine_grained_label'] = mapped_predictions[idx_in_list]

    reverse_label_map = {v: k for k, v in label_map.items()}
    test_data['fine_grained_label_name'] = test_data['fine_grained_label'].map(reverse_label_map)


    test_data.to_csv('combined_predictions.csv', index=False)
    print("Combined predictions saved to 'combined_predictions.csv'.")

    for idx, row in test_data.head(10).iterrows():
        print(f"Text: {row['text']}")
        print(f"Coarse Label Prediction: {'Request' if row['coarse_label_pred'] == 1 else 'Non-Request'}")
        if row['coarse_label_pred'] == 1:
            print(f"Fine-Grained Label: {row['fine_grained_label_name']}")
        print("---")

#if __name__ == "__main__":
#    main()
