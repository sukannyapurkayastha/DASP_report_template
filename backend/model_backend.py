# model_backend.py

import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

label_mapping = {
    0: 'Other',
    1: 'Clarity',
    2: 'Meaningful-comparison',
    3: 'Motivation-impact',
    4: 'Originality',
    5: 'Replicability',
    6: 'Soundness-correctness',
    7: 'Substance',
    8: 'None'
}

# **Tokenizer und Modell einmalig laden**
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=9)

def predict_category(text):
    # Verwenden des bereits geladenen Tokenizers und Modells
    predict_input = tokenizer.encode(text,
                                     truncation=True,
                                     padding=True,
                                     return_tensors="tf")
    output = model(predict_input)[0]
    prediction_value = tf.argmax(output, axis=1).numpy()[0]
    return prediction_value

def classify_paper(data):
    # 'data' ist eine Liste von Dictionaries mit Schlüsseln 'sentence' und 'author'
    try:
        review_sents = pd.DataFrame(data)
        review_sents['attitude_root_number'] = review_sents['sentence'].apply(predict_category)
        total_rows = len(review_sents)

        # Gruppieren und Aggregieren
        agg_df = review_sents.groupby(['attitude_root_number', 'author']).agg(
            comments=('sentence', list), count=('sentence', 'size')
        ).reset_index()

        final_df = agg_df.groupby('attitude_root_number').agg(
            comments=('comments', lambda x: [[author, comments] for author, comments in zip(agg_df['author'], x)]),
            count=('count', 'sum')
        ).reset_index()

        # DataFrame anpassen
        final_df['Attitude_roots'] = final_df['attitude_root_number'].map(label_mapping)
        final_df['Descriptions'] = 'none'
        final_df.drop(columns=['attitude_root_number'], inplace=True)
        final_df = final_df.rename(columns={'comments': 'Comments'})
        final_df['Frequency'] = final_df['count'] / total_rows
        final_df.drop(columns=['count'], inplace=True)
        final_df = final_df[['Attitude_roots', 'Frequency', 'Descriptions', 'Comments']]

        # Optional: DataFrame als Pickle speichern
        # final_df.to_pickle('data.pkl')

        # DataFrame zurückgeben
        return final_df

    except Exception as e:
        print(f"Error processing paper: {str(e)}")
        return None
