from typing import Any, List, Generator
import pandas as pd
import uvicorn
from loguru import logger
from pydantic import BaseModel
from text_processing import Review, TextProcessor


def process_file(reviews_json: list[dict]) -> (pd.DataFrame, pd.DataFrame):  # change amount of df to 4 (or 5)
    """
    Processes a list of reviews. This means that each review will be first sentencized and then be given to nlp models to classify the sentences.
    :param reviews: list of reviews (from frontend)
    :return: 4 different pandas dataframes (1 from the review content directly, and 3 dataframes from the model output)
    """

    reviews = [Review.from_dict(review_dict) for review_dict in reviews_json]
    text_processer = TextProcessor(reviews=reviews)
    df_sentences, df_overview = text_processer.process()  # df_sentences is the input for the models, df_overview is to be directly sent to the frontend

    return df_overview, df_sentences
