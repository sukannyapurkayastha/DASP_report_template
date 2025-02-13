import re

import pandas as pd
import spacy
from tqdm import tqdm
from spacy.language import Language
from spacy.tokens.doc import Doc
from loguru import logger

from .models import Review


# SpaCy component to split only at linebreaks
@Language.component("split_only_at_linebreaks")
def split_only_at_linebreaks(doc: Doc) -> Doc:
    """
    Custom SpaCy component to split sentences only at line breaks.

    :param doc: SpaCy Doc object
    :return: Processed Doc object with sentence boundaries set at line breaks only.
    """ 
    for token in doc:
        token.is_sent_start = False

    for i, token in enumerate(doc[:-1]):
        if token.text == "\n" or token.text == "\n ":
            if i + 1 < len(doc):
                doc[i + 1].is_sent_start = True
    return doc


class TextProcessor:
    """
    A class for processing and analyzing textual review data. This includes preprocessing, segmentation,
    and generating statistical overviews.
    """
    def __init__(self, reviews: list[Review]):
        """
        Initializes the TextProcessor with a list of Review objects.

        :param reviews: List of Review objects to be processed.
        """
        self.reviews = reviews
        self.config = {"punct_chars": ['\n']}
        self.nlp = spacy.load("en_core_web_sm", exclude=["parser"])
        self.nlp.add_pipe("sentencizer")
        self.nlp.add_pipe("split_only_at_linebreaks", before="sentencizer")
        self.keys_to_extract = ["summary", "strengths", "weaknesses", "questions"]
        self.overview_keys = ["rating", "soundness", "presentation", "contribution"]

        self.abbreviations = [
            "e.g.", "i.e.", "Dr.", "Mr.", "Mrs.", "Prof.", "etc.", "Fig.", "Tab.", "al.", "et al.", "vol.", "no.",
            "pp.", "Ch.", "Eq.", "Sec.", "Ref.", "Inc.", "Corp.", "Ltd.", "Co.", "Vs.", "approx.", "esp.", "incl.",
            "viz.", "resp.", "cf.", "avg.", "max.", "min.", "std.", "dev.", "var.", "mean.", "median.", "p-value",
            "s.d.", "a.k.a.", "b.c.", "a.d.", "m.a.", "n.d.", "ca."]

    @staticmethod
    def _protect_abbreviation(abbreviations: list[str], text: str) -> str:
        """
        Replaces periods in abbreviations with a placeholder to avoid unintended sentence splitting.

        :param abbreviations: List of abbreviations to protect.
        :param text: Text in which abbreviations should be protected.
        :return: Processed text with protected abbreviations.
        """
        for abbr in abbreviations:
            protected = abbr.replace(".", "<DOT>")
            text = text.replace(abbr, protected)
        return text

    def _preprocess_text(self, reviews: list[Review]) -> list[Review]:
        """
        Cleans and normalizes text fields within the review objects.

        :param reviews: List of Review objects.
        :return: List of processed Review objects with cleaned text fields.
        """
        for review in reviews:
            for key in self.keys_to_extract:
                text = getattr(review, key, "")
                text = text.strip()
                text = "\n".join(" ".join(line.split()) for line in text.splitlines())
                text = text.encode("utf-8", "ignore").decode("utf-8")
                text = re.sub(r"http\S+", "", text)
                text = re.sub(r"\S+@\S+", "", text)
                text = text.replace(";", ".")
                text = text.replace("*", "")
                text = re.sub(r"(?<![.!?])\n", " ", text)
                text = self._protect_abbreviation(self.abbreviations, text)
                text = text.strip()
                text = text + "\n"
                setattr(review, key, text)
        return reviews

    def _segment_content(self, reviews: list[Review]) -> (list[Review], pd.DataFrame):
        """
        Segments text content into sentences using SpaCy and stores them in a DataFrame.

        :param reviews: List of Review objects.
        :return: Tuple containing the updated list of reviews and a DataFrame with segmented sentences.
        """
        sent_df = []
        for review in tqdm(reviews, desc="Segmenting content"):
            for key in self.keys_to_extract:
                text = getattr(review, key, "")
                doc = self.nlp(text)
                sentences = [sent.text.replace("<DOT>", ".") for sent in doc.sents]
                sentences = [" ".join(sent.split()) for sent in sentences]
                sentences = [sent.lstrip("-").strip() for sent in sentences]
                review.sentences = sentences
                tmp = pd.DataFrame({"author": review.reviewer, "tag": key, "sentence": sentences})
                sent_df.append(tmp)

        df = pd.concat(sent_df, ignore_index=True)
        return reviews, df

    def _get_overview(self, reviews: list[Review]) -> pd.DataFrame:
        """
        Computes an overview of review scores.

        :param reviews: List of Review objects.
        :return: DataFrame summarizing average scores per category.
        """
        df_list = []

        for key in self.overview_keys:
            score_list = []
            total = 0
            for review in reviews:
                reviewer = review.reviewer
                tmp = getattr(review, key, "")
                if isinstance(tmp, str):
                    score = float(tmp.split(" ")[0].replace(":", ""))
                else:
                    score = float(tmp)
                total += score
                list = [reviewer, score]
                score_list.append(list)
            avg = total / len(score_list)
            df_list.append({"Category": key.title(), "Avg_Score": avg, "Individual_scores": score_list})

        return pd.DataFrame(df_list)

    def process(self) -> (pd.DataFrame, pd.DataFrame):
        """
        Processes and segments the reviews. Creates a sentences dataframe aan overview dataframe of the scores .
        :return: Sentences dataframe and overview dataframe
        """
        logger.info(f"Processing reviews")
        preprocessed_reviews = self._preprocess_text(self.reviews)
        logger.info("Segmenting content")
        reviews, df_sentences = self._segment_content(preprocessed_reviews)
        logger.info("Got reviews and df_sentences. Creating overview.")
        df_overview = self._get_overview(preprocessed_reviews)

        logger.success(f"Processing reviews done")
        return df_sentences, df_overview
