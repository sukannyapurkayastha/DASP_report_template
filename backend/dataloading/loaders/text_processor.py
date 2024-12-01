import re

import pandas as pd
import spacy
from tqdm import tqdm
from spacy.language import Language
from spacy.tokens.doc import Doc

from backend.dataloading.loaders.models import Review


# SpaCy components to split only at linebreaks and to protect lists and citations
# Component 1: only split at line breaks
@Language.component("split_only_at_linebreaks")
def split_only_at_linebreaks(doc: Doc) -> Doc:
    for token in doc:
        token.is_sent_start = False

    for i, token in enumerate(doc[:-1]):
        if token.text == "\n" or token.text == "\n ":
            if i + 1 < len(doc):
                doc[i + 1].is_sent_start = True
    return doc


# Component 2: Prevent splitting at enumerations like "1.", "2.", "1.1."
@Language.component("prevent_splitting_on_list_numbers")
def prevent_splitting_on_list_numbers(doc: Doc) -> Doc:
    for i, token in enumerate(doc[:-2]):
        # Check if the token is a number followed by a period
        if token.like_num and doc[i + 1].text == ".":
            if i + 2 < len(doc):
                doc[i + 2].is_sent_start = False
    return doc


# Component 3: Prevent splitting after citations like "[2]."
@Language.component("prevent_splitting_on_citations")
def prevent_splitting_on_citations(doc: Doc) -> Doc:
    def is_citation(tokens, end_idx):
        idx = end_idx
        if tokens[idx].text != "]":
            return False
        idx -= 1
        # Collect tokens that are numbers, commas, or hyphens (for ranges)
        while idx >= 0 and (tokens[idx].like_num or tokens[idx].text in [",", "-", "–"]):
            idx -= 1
        if idx >= 0 and tokens[idx].text == "[":
            return True
        else:
            return False

    for i, token in enumerate(doc[:-2]):
        # Check for citations ending with "]."
        if token.text == "]" and doc[i + 1].text == ".":
            if is_citation(doc, i):
                if i + 2 < len(doc):
                    doc[i + 2].is_sent_start = False
    return doc


class TextProcessor:
    def __init__(self):
        self.config = {"punct_chars": ['\n']}
        self.nlp = spacy.load("en_core_web_sm", exclude=["parser"])
        self.nlp.add_pipe("sentencizer")
        self.nlp.add_pipe("split_only_at_linebreaks", before="sentencizer")
        # self.nlp.add_pipe("prevent_splitting_on_list_numbers", before="sentencizer")
        # self.nlp.add_pipe("prevent_splitting_on_citations", before="sentencizer")

        self.abbreviations = [
            "e.g.", "i.e.", "Dr.", "Mr.", "Mrs.", "Prof.", "etc.", "Fig.", "Tab.", "al.", "et al.", "vol.", "no.",
            "pp.", "Ch.", "Eq.", "Sec.", "Ref.", "Inc.", "Corp.", "Ltd.", "Co.", "Vs.", "approx.", "esp.", "incl.",
            "viz.", "resp.", "cf.", "avg.", "max.", "min.", "std.", "dev.", "var.", "mean.", "median.", "p-value",
            "s.d.", "a.k.a.", "b.c.", "a.d.", "m.a.", "n.d.", "ca."]

    @staticmethod
    def _protect_abbreviation(abbreviations: list[str], text: str) -> str:
        for abbr in abbreviations:
            protected = abbr.replace(".", "<DOT>")
            text = text.replace(abbr, protected)
        return text

    def _preprocess_text(self, reviews: list[Review],
                         keys_to_extract: list[str] = ["summary", "strengths", "weaknesses", "questions"]) -> list[
        Review]:
        for review in reviews:
            content_list = []
            for key in keys_to_extract:
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
                content_list.append(text)
            review.content = " ".join(content_list)
        return reviews

    def _segment_content(self, reviews: list[Review]) -> (list[Review], pd.DataFrame):
        sent_df = []
        for review in tqdm(reviews, desc="Segmenting content"):
            text = review.content
            doc = self.nlp(text)
            sentences = [sent.text.replace("<DOT>", ".") for sent in doc.sents]
            sentences = [" ".join(sent.split()) for sent in sentences]
            sentences = [sent.lstrip("-").strip() for sent in sentences]
            review.sentences = sentences
            tmp = pd.DataFrame({"author": review.reviewer, "sentence": sentences})
            sent_df.append(tmp)

        df = pd.concat(sent_df, ignore_index=True)
        return reviews, df
