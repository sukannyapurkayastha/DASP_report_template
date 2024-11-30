import re

import openreview
import pandas as pd
from loguru import logger
from typing import Literal
import spacy
from spacy.language import Language
from spacy.tokens.doc import Doc

from dataclasses import dataclass

from tqdm import tqdm


# SpaCy pipeline to protect lists and citations
# Component 1: Prevent splitting at enumerations like "1.", "2.", "1.1."
@Language.component("prevent_splitting_on_list_numbers")
def prevent_splitting_on_list_numbers(doc: Doc) -> Doc:
    for i, token in enumerate(doc[:-2]):
        # Check if the token is a number followed by a period
        if token.like_num and doc[i + 1].text == ".":
            if i + 2 < len(doc):
                doc[i + 2].is_sent_start = False
    return doc


# Component 2: Prevent splitting after citations like "[2]."
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


# Component to prevent splitting on all other punctuation except line breaks
@Language.component("split_only_at_linebreaks")
def split_only_at_linebreaks(doc: Doc) -> Doc:
    for token in doc:
        token.is_sent_start = False

    for i, token in enumerate(doc[:-1]):
        if token.text == "\n" or token.text == "\n ":
            if i + 1 < len(doc):
                doc[i + 1].is_sent_start = True
    return doc


@dataclass
class Review:
    forum: str
    id: str
    venue: str
    year: str
    type: str
    reviewer: str
    date: str | pd.Timestamp
    rating: str | int
    soundness: str | int
    presentation: str | int
    contribution: str | int
    summary: str
    strengths: str
    weaknesses: str
    questions: str
    flag_for_ethic_review: str
    confidence: str
    content: str | None = None  # All the text in a single string
    sentences: list[str] | None = None  # Output of spacy sentencizer

@dataclass
class Paper:
    title: str
    authors: list[str]
    keywords: list[str]
    abstract: str
    primary_area: str
    venue: str
    cdate: int
    domain: str
    forum: str
    id: str
    directReplies: list[dict]
    reviews: list[Review] | None = None
    sentences: list[str] | None = None


class OpenReviewLoader(object):
    def __init__(
            self,
            username: str,
            password: str,
            baseurl: str = "https://api2.openreview.net",
    ):
        """
        Initialize a Wrapper around the OpenReview client to load reviews, segment them, and create testsets.

        :param username: OpenReview username
        :param password: OpenReview password
        :param baseurl: The base url for the OpenReview API v2.
        """
        self.client = openreview.api.OpenReviewClient(
            baseurl=baseurl,
            username=username,
            password=password,
        )
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

        logger.info("OpenReview client initialized.")

    @staticmethod
    def _is_comment_by_author(signatures: list) -> bool:
        return any("Authors" in signature for signature in signatures)

    @staticmethod
    def _is_meta_review(signatures: list) -> bool:
        return any("Senior_Area_Chairs" in signature or "Area_Chair" in signature for signature in signatures)

    @staticmethod
    def _is_review(invitations: list) -> bool:
        return any("Official_Review" in invitation for invitation in invitations)

    @staticmethod
    def _is_paper_decision(note: dict) -> bool:
        return ("title" in note["content"]) & ("decision" in note["content"])

    @staticmethod
    def _is_paper(note: openreview.Note) -> bool:
        return ("title" in note.content) & ("authors" in note.content)

    @staticmethod
    def _protect_abbrevation(abbrevations: list[str], text: str) -> str:
        for abbr in abbrevations:
            protected = abbr.replace(".", "<DOT>")
            text = text.replace(abbr, protected)

        return text

    def load_all_submissions(self, venue: str, year: int, type: str = "Conference",
                             details: Literal["replies", "directReplies"] = "directReplies") -> list[openreview.Note]:
        """
        Load all submissions from a specific venue.
        :param venue: The venue name (e.g. "ICLR.cc")
        :param year: The year of the venue (e.g. 2024)
        :param type: The type of venue (e.g. "Conference", "TinyPapers)
        :param details: The detail of retrieved information (e.g. "replies", "directReplies")
        :return: A list of submissions
        """
        invitation = f"{venue}/{year}/{type}/-/Submission"
        logger.info(f"Fetching submissions with invitation: {invitation}")

        try:
            submissions = self.client.get_all_notes(invitation=invitation, details=details)
            logger.info(f"Fetched {len(submissions)} submissions")
            return submissions
        except Exception as e:
            logger.error("Failed to fetch submissions from OpenReview.")
            logger.error(e)
            return []

    def get_paper(self, id: str) -> Paper:
        """
        Load an individual paper.
        :param id: The id of the paper.
        :return: Information on the paper.
        """
        logger.info(f"Fetching paper with id: {id}")

        try:
            papers = self.client.get_all_notes(forum=id, details="directReplies")
            _paper = next((papers.pop(idx) for idx, reply in enumerate(papers) if self._is_paper(reply)), None)
            paper = Paper(
                title=_paper.content["title"]["value"],
                authors=_paper.content["authors"]["value"],
                keywords=_paper.content["keywords"]["value"],
                abstract=_paper.content["abstract"]["value"],
                primary_area=_paper.content["primary_area"]["value"],
                venue=_paper.content["venue"]["value"],
                cdate=_paper.cdate,
                domain=_paper.domain,
                forum=_paper.forum,
                id=_paper.id,
                directReplies=_paper.details["directReplies"],
            )
            logger.info(f"Fetched paper: {paper.title}")
            return paper
        except Exception as e:
            logger.error("Failed to fetch paper from OpenReview.")
            logger.error(e)
            return None

    def prepare_reviews(self, notes: list[dict]) -> list[Review]:
        """
        Prepares a list of reviews
        :param notes: The openreview Notes to prepare.
        :return: A list of prepared reviews
        """
        prepared_reviews = []
        for note in notes:
            prepared_reviews.append(self.prepare_review(note))

        return prepared_reviews

    def prepare_review(self, note: dict) -> Review:
        """
        Prepares a review.
        :param note: The openreview Note as a dict to prepare.
        :return: A prepared review.
        """

        logger.info(f"Preparing review for note: {note['id']}")
        venue, year, type = note["domain"].split("/")
        reviewer = next((item.split("/")[-1] for item in note["writers"] if "Reviewer_" in item), None)

        try:
            content_dict = {}
            for key, val in note["content"].items():
                if isinstance(val, dict) and "value" in val:
                    if isinstance(val["value"], list):
                        text_value = " ".join(map(str, val["value"]))
                    else:
                        text_value = val["value"]
                    content_dict[key] = text_value
                else:
                    content_dict[key] = val
        except:
            exit("Review has no content or is of wrong format.")

        review = Review(
            forum=note["forum"],
            id=note["id"],
            venue=venue,
            year=year,
            type=type,
            reviewer=reviewer.replace("_", " "),
            date=note["cdate"],
            rating=content_dict["rating"],
            soundness=content_dict["soundness"],
            presentation=content_dict["presentation"],
            contribution=content_dict["contribution"],
            summary=content_dict["summary"],
            strengths=content_dict["strengths"],
            weaknesses=content_dict["weaknesses"],
            questions=content_dict["questions"],
            flag_for_ethic_review=content_dict["flag_for_ethics_review"],
            confidence=content_dict["confidence"]
        )

        return review

    def _get_reviews(self, paper: Paper) -> list[Review]:
        """
        Load all reviews from a specific paper.
        :param paper: The paper.
        :return: All official reviews for a paper.
        """

        logger.info(f"Getting reviews for paper: {paper.title}")

        try:
            replies = paper.directReplies
            replies = [reply for reply in replies if self._is_review(reply["invitations"])]
            # The following tests are actually no longer needed as far as I can judge
            _ = [replies.pop(idx) for idx, reply in enumerate(replies) if self._is_comment_by_author(reply["writers"])]
            _ = [replies.pop(idx) for idx, reply in enumerate(replies) if self._is_meta_review(reply["writers"])]
            _ = [replies.pop(idx) for idx, reply in enumerate(replies) if self._is_paper_decision(reply)]
            prepared_reviews = self.prepare_reviews(replies)
            return prepared_reviews
        except Exception as e:
            logger.error("Failed to fetch paper reviews from OpenReview.")
            logger.error(e)
            return []

    def _get_reviews_from_multiple_papers(self, ids: list[str]) -> list:
        """
        Load all reviews from multiple papers.
        :param ids: The ids of the papers.
        :return: All reviews for multiple papers.
        """
        logger.info(f"Fetching reviews from multiple papers")

        reviews = []
        for id in ids:
            paper = self.get_paper(id)
            single_paper_reviews = self._get_reviews(paper)
            reviews.extend(single_paper_reviews)

        return reviews

    def _preprocess_text(self, reviews: list[Review],
                         keys_to_extract: list[str] = ["summary", "strengths", "weaknesses", "questions"]) -> list[
        Review]:
        """
        Process the reviews provided as a list of strings.
        :param reviews: List of openreview notes.
        :param keys_to_extract: Only alphanumeric comment fields are of interest
        :return: List of processed reviews.
        """

        # abbreviations = ["e.g.", "i.e.", "Dr.", "Mr.", "Mrs.", "etc."]

        for review in reviews:
            content_list = []
            for key in keys_to_extract:
                text = getattr(review, key)
                text = text.strip()  # Remove leading and trailing whitespaces
                text = "\n".join(" ".join(line.split()) for line in text.splitlines())  # Whitespaces in the string
                text = text.encode("utf-8", "ignore").decode("utf-8")  # Handle special characters
                text = re.sub(r"http\S+", "", text)  # Remove URLs
                text = re.sub(r"\S+@\S+", "", text)  # Remove email addresses
                text = text.replace(";", ".")  # Replace non-standard sentence delimiters
                text = text.replace("*", "")  # Remove asterisk (are used for bullet list tokens)
                text = re.sub(r"(?<![.!?])\n", " ", text)
                text = self._protect_abbrevation(self.abbreviations, text)

                text = text.strip()
                content_list.append(text)
            review.content = " ".join(content_list)

        return reviews

    def _segment_content(self, reviews: list[Review]) -> (list[Review], list[str]):
        """
        Segment the content of the reviews provided into single sentences.
        :param reviews: The preprocessed reviews
        :return: 2 objects, List of all reviews with the variable .sentences set to the output of the spacy sentencizer and second object is a list of sentences.
        """

        sents = []

        for review in tqdm(reviews):
            text = review.content
            doc = self.nlp(text)

            sentences = [sent.text.replace("<DOT>", ".") for sent in doc.sents]
            sentences = [" ".join(sent.split()) for sent in sentences]
            sentences = [sent.lstrip("-").strip() for sent in sentences]

            review.sentences = sentences
            sents.extend(sentences)

        return reviews, sents

    def create_testset(self, ids: list[str]) -> list[str]:
        """
        Create a test set from multiple paper reviews.
        :param ids: The ids of the papers.
        :return: List with reviews segmented in sentences.
        """

        reviews = self._get_reviews_from_multiple_papers(ids)
        processed_reviews = self._preprocess_text(reviews)  # already preprocessed
        final_reviews, sentences = self._segment_content(processed_reviews)

        return sentences

    def get_reviews(self, id: str | list[str]) -> list[Review]:
        """
        Load all reviews from a specific paper.
        :param id: The id of the paper.
        :return: All reviews for a paper.
        """

        if isinstance(id, str):
            paper = self.get_paper(id)
            reviews = self._get_reviews(paper)
        else:
            reviews = self._get_reviews_from_multiple_papers(ids=id)
        processed_reviews = self._preprocess_text(reviews)
        final_reviews, sentences = self._segment_content(processed_reviews)

        return final_reviews

    def get_paper_reviews(self, id: str) -> Paper:
        """
        Load all reviews from a specific paper.
        :param id: The id of the paper
        :return: Paper object
        """
        paper = self.get_paper(id)
        reviews = self._get_reviews(paper)
        processed_reviews = self._preprocess_text(reviews)
        paper.reviews, paper.sentences = self._segment_content(processed_reviews)

        return paper

    def get_all_submission_reviews(self, venue: str, year: int, type: str = "Conference",
                                   details: Literal["replies", "directReplies"] = "directReplies") -> list[Review]:
        """
        Load all submissions from a specific venue.
        :param venue: The venue name (e.g. "ICLR.cc")
        :param year: The year of the venue (e.g. 2024)
        :param type: The type of venue (e.g. "Conference", "TinyPapers)
        :param details: The detail of retrieved information (e.g. "replies", "directReplies")
        :return: A list of submissions
        """

        submission = self.load_all_submissions(venue=venue, year=year, type=type)

        reviews = []
        for paper in submission:
            single_paper_reviews = self._get_reviews(paper)
            reviews.extend(single_paper_reviews)

        processed_reviews = self._preprocess_text(reviews)
        final_reviews, sentences = self._segment_content(processed_reviews)

        return final_reviews
