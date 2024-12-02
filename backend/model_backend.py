from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import pandas as pd
import uvicorn

app = FastAPI()

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

# Create a Pydantic model to handle the input data
class SentenceData(BaseModel):
    data: list[Dict]

def predict_category(text):
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('models/attitude_root/')

    # Load the model
    model = TFBertForSequenceClassification.from_pretrained('models/attitude_root/')
    predict_input = tokenizer.encode(text,
                                    truncation=True,
                                    padding=True,
                                    return_tensors="tf")
    output = model(predict_input)[0]
    prediction_value = tf.argmax(output, axis=1).numpy()[0]
    return prediction_value

@app.post("/roots_themes")
async def classify_paper(data: SentenceData):
    try:
        review_sents = data.data
        review_sents['attitude_root_number'] = review_sents['sentence'].apply(predict_category)
        total_rows = len(review_sents)
        # Step 1: Group by 'attitude_root_number' and 'author', then aggregate the sentences
        agg_df = review_sents.groupby(['attitude_root_number', 'author']).agg(
                    comments=('sentence', list), count=('sentence', 'size')
                ).reset_index()
        final_df = agg_df.groupby('attitude_root_number').agg(
                    comments=('comments', lambda x: [[author, comments] for author, comments in zip(agg_df['author'], x)]),
                    count=('count', 'sum')
                ).reset_index()
        final_df = final_df['attitude_root_number'].map(label_mapping)
        final_df['description'] = 'none'
        # Convert DataFrame to a list of dictionaries
        results = final_df.to_dict(orient='records')
        # Return the tabular data as a JSON response
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing paper: {str(e)}")

