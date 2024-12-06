from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
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
    tokenizer = DistilBertTokenizer.from_pretrained('models/attitude_root/')

    # Load the model
    model = TFDistilBertForSequenceClassification.from_pretrained('models/attitude_root/')
    predict_input = tokenizer.encode(text,
                                    truncation=True,
                                    padding=True,
                                    return_tensors="tf")
    output = model(predict_input)[0]
    prediction_value = tf.argmax(output, axis=1).numpy()[0]
    return prediction_value

@app.post("/roots_themes")
async def classify_paper(data: SentenceData):
    # right now async not necessary
    try:
        review_sents = pd.DataFrame(data.data)
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
        
        # match dataframe schema
        final_df['Attitude_roots'] = final_df['attitude_root_number'].map(label_mapping)
        final_df['Descriptions'] = 'none'
        final_df.drop(columns=['attitude_root_number'], inplace=True)
        final_df = final_df.rename(columns={'comments': 'Comments'})
        final_df['Frequency'] = final_df['count'] / total_rows
        final_df.drop(columns=['count'], inplace=True)
        final_df = final_df[['Attitude_roots', 'Frequency', 'Descriptions', 'Comments']]
        final_df = final_df.sort_values(by='Frequency', ascending=False)


        desc = pd.read_csv(r"..\..\data\attitude_roots\attitudes_desc.csv")

        merged_df = pd.merge(final_df , desc, on=['Attitude_roots'], how='left') # todo:what happens if attitude + theme combi is not known
        merged_df.rename(columns={'Descriptions_y': 'Descriptions'}, inplace=True)

        # Drop Descriptions_x column
        merged_df.drop(columns=['Descriptions_x'], inplace=True)
        merged_df = merged_df[['Attitude_roots', 'Frequency', 'Descriptions', 'Comments']]

        # Convert DataFrame to a list of dictionaries
        results = merged_df.to_dict(orient='records')
        # Return the tabular data as a JSON response
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing paper: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)