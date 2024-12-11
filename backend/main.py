from typing import Any, List, Generator

from fastapi import FastAPI
import pandas as pd
import uvicorn
from loguru import logger
from pydantic import BaseModel

from text_processing import Review, TextProcessor
from models.request_classifier.request_classifier_pipeline import process_dataframe_request

app = FastAPI()


class ReviewData(BaseModel):
    data: list[dict]

#     class Config:
#         arbitrary_types_allowed = True
#
# class ProcessesOutput(BaseModel):
#     df_sentences: pd.DataFrame
#     df_overview: pd.DataFrame
#
#     class Config:
#         arbitrary_types_allowed = True


@app.post("/process")
async def process_file(reviews_json: ReviewData) -> dict:  # change amount of df to 4 (or 5)
    """
    Processes a list of reviews. This means that each review will be first sentencized and then be given to nlp models to classify the sentences.
    :param reviews: list of reviews (from frontend)
    :return: 4 different pandas dataframes (1 from the review content directly, and 3 dataframes from the model output)
    """
    logger.info("Backend :)")
    try:
        data = reviews_json.data

        reviews = [Review.from_dict(review_dict) for review_dict in data]
        text_processer = TextProcessor(reviews=reviews)
        print("Establish TextProcessor")
        df_sentences, df_overview = text_processer.process()  # df_sentences is the input for the models, df_overview is to be directly sent to the frontend
        df_request = process_dataframe_request(df_sentences)

        df_sentences.to_csv("backend/df_sentences.csv", index=False)
        # Todo: Write functions in which each model is loaded and df_sentences is given as input, return the model output
        # Todo: The model output (dataframe) should then be cheanged to a dict and added to the return dict
        # df_classifier_request = request_classifier(df_sentences)

        # Convert DataFrames to JSON-serializable formats
        df_request_json = df_sentences.to_dict(orient='records')
        df_sentences_json = df_sentences.to_dict(orient='records')
        df_overview_json = df_overview.to_dict(orient='records')

        # return ProcessesOutput(df_sentences=df_sentences, df_overview=df_overview)

        # Todo: Remove df_senteces, it's just to test multiple dataframes
        return {
            "df_request": df_sentences_json,
            "df_sentences": df_sentences_json,
            "df_overview": df_overview_json
        }
        # return None
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
