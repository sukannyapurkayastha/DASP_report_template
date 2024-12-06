from typing import Any, List, Generator
import pandas as pd
import uvicorn
from loguru import logger
from pydantic import BaseModel
from text_processing import Review, TextProcessor

class ReviewData(BaseModel):
    data: list[dict]

def process_file(reviews_json: ReviewData):  # change amount of df to 4 (or 5)
    """
    Processes a list of reviews. This means that each review will be first sentencized and then be given to nlp models to classify the sentences.
    :param reviews: list of reviews (from frontend)
    :return: 4 different pandas dataframes (1 from the review content directly, and 3 dataframes from the model output)
    """
    # logger.info("Backend :)")
    data = reviews_json.data

    for idx, x in enumerate(data):
        print(f"idx {idx}: Type {type(x)}")

    reviews = [Review.from_dict(review_dict) for review_dict in data]
    text_processer = TextProcessor(reviews=reviews)
    # print("Establish TextProcessor")
    df_sentences, df_overview = text_processer.process()  # df_sentences is the input for the models, df_overview is to be directly sent to the frontend

    # Todo: Write functions in which each model is loaded and df_sentences is given as input, return the model output
    # Todo: The model output (dataframe) should then be cheanged to a dict and added to the return dict

    # # Convert DataFrames to JSON-serializable formats
    # df_sentences_json = df_sentences.to_dict(orient='records')
    # df_overview_json = df_overview.to_dict(orient='records')

    return df_overview, df_sentences
    # return None


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080)
