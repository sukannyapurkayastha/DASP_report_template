import pandas as pd
from text_processing import Review, TextProcessor


def process_file(reviews_json: list[dict]) -> (pd.DataFrame, pd.DataFrame):
    """
    Processes a list of reviews. This means that each review will be first sentencized and then be given to nlp models
    to classify the sentences.
    :param reviews: list of reviews (from frontend)
    :return: input for request and attitude models and overview for frontend visualization
    """

    reviews = [Review.from_dict(review_dict) for review_dict in reviews_json]
    text_processer = TextProcessor(reviews=reviews)
    df_sentences, df_overview = text_processer.process()

    return df_overview, df_sentences
