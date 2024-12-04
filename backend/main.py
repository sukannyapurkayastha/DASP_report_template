from typing import Any

from fastapi import FastAPI
import pandas as pd
import uvicorn
from loguru import logger
from pydantic import BaseModel

from text_processing import Review, TextProcessor

app = FastAPI()

class ReviewData(BaseModel):
    data: list[dict[str, Any]]


@app.post("/process/")
async def process_file(reviews: ReviewData) -> dict:  # change amount of df to 4 (or 5)
    """
    Processes a list of reviews. This means that each review will be first sentencized and then be given to nlp models to classify the sentences.
    :param reviews: list of reviews (from frontend)
    :return: 4 different pandas dataframes (1 from the review content directly, and 3 dataframes from the model output)
    """
    logger.info("Backend :)")
    try:
        data = reviews.data
        print(data)
        text_processer = TextProcessor(reviews=reviews.data)
        df_sentences, df_overview = text_processer.process()  # df_sentences is the input for the models, df_overview is to be directly sent to the frontend

        # Convert DataFrames to JSON-serializable formats
        df_sentences_json = df_sentences.to_dict(orient='records')
        df_overview_json = df_overview.to_dict(orient='records')

        return {
            "df_sentences": df_sentences_json,
            "df_overview": df_overview_json
        }
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
