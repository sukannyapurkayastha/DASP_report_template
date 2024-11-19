import re

import openreview
import pandas as pd
from loguru import logger
from typing import List, Literal
import spacy

from dataclasses import dataclass


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
        self.nlp = spacy.load('en_core_web_sm')

        logger.info("OpenReview client initialized.")

    @staticmethod
    def _review_by_author(signatures: list) -> bool:
        return any("Authors" in signature for signature in signatures)
    
    @staticmethod
    def _is_meta_review(signatures: list) -> bool:
        return any("Senior_Area_Chairs" in signature or "Area_Chair" in signature for signature in signatures)

    @staticmethod
    def _is_paper_decision(note: dict) -> bool:
        return ("title" in note["content"]) & ("decision" in note["content"])

    @staticmethod
    def _is_paper(note: openreview.Note) -> bool:
        return ("title" in note.content) & ("authors" in note.content)

    def load_all_submissions(self, venue: str, year: int, type: str = "Conference",
                             detail: Literal["replies", "directReplies"] = "directReplies") -> List[openreview.Note]:
        """
        Load all submissions from a specific venue.
        :param venue: The venue name (e.g. "ICLR.cc")
        :param year: The year of the venue (e.g. "2024")
        :param type: The type of venue (e.g. "Conference", "TinyPapers)
        :param detail: The detail of retrieved information (e.g. "replies", "directReplies")
        :return: A list of submissions
        """
        invitation = f"{venue}/{year}/{type}/-/Submission"
        logger.info(f"Fetching submissions with invitation: {invitation}")

        try:
            submissions = self.client.get_all_notes(invitation=invitation, detail=detail)
            logger.info(f"Fetched {len(submissions)} submissions")
            return submissions
        except Exception as e:
            logger.error("Failed to fetch submissions from OpenReview.")
            logger.error(e)
            return []

    def get_paper(self, id: str) -> openreview.Note:
        """
        Load an individual paper.
        :param id: The id of the paper.
        :return: Information on the paper.
        """
        logger.info(f"Fetching paper with id: {id}")

        try:
            papers = self.client.get_all_notes(forum=id, details="directReplies")
            paper = next((papers.pop(idx) for idx, reply in enumerate(papers) if self._is_paper(reply)), None)
            logger.info(f"Fetched paper: {paper.content['title']['value']}")
            return paper
        except Exception as e:
            logger.error("Failed to fetch paper from OpenReview.")
            logger.error(e)
            return []

    def prepare_reviews(self, notes: List[openreview.Note]) -> List[Review]:
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
        reviewer = next((item.split('/')[-1] for item in note["writers"] if 'Reviewer_' in item), None)

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

    def get_reviews(self, id: str) -> List[Review]:
        """
        Load all reviews from a specific paper.
        :param id: The id of the paper.
        :return: All reviews for a paper.
        """

        logger.info(f"Fetching reviews for paper with id: {id}")

        try:
            paper = self.get_paper(id=id)
            replies = paper.details["directReplies"].copy()
            _ = [replies.pop(idx) for idx, reply in enumerate(replies) if self._review_by_author(reply["writers"])]
            _ = [replies.pop(idx) for idx, reply in enumerate(replies) if self._is_meta_review(reply["writers"])]
            _ = [replies.pop(idx) for idx, reply in enumerate(replies) if self._is_paper_decision(reply)]
            prepared_reviews = self.prepare_reviews(replies)
            return prepared_reviews
        except Exception as e:
            logger.error("Failed to fetch paper reviews from OpenReview.")
            logger.error(e)
            return []

    def get_reviews_from_multiple_papers(self, ids: List[str]) -> List:
        """
        Load all reviews from multiple papers.
        :param ids: The ids of the papers.
        :return: All reviews for multiple papers.
        """
        logger.info(f"Fetching reviews from multiple papers")

        reviews = []
        for id in ids:
            single_paper_reviews = self.get_reviews(id)
            reviews.extend(single_paper_reviews)

        return reviews

    def _process_reviews(self, reviews: List[openreview.Note],
                         keys_to_extract: List[str] = ["summary", "strengths", "weaknesses", "questions", "rating",
                                                       "confidence"]) -> List[str]:
        """
        Process the reviews provided as a list of strings.
        :param reviews: List of openreview notes.
        :param keys_to_extract: Only alphanumeric comment fields are of interest
        :return: List of processed reviews.
        """

        review_contents = []

        for review in reviews:
            replies = review.details["directReplies"]
            for reply in replies:
                # Only get replies/comments/reviews from reviewers
                if not "Reviewer" in reply["writers"][1]:
                    continue
                content = reply["content"]

                # Undo nested dictionary structure
                new_content_dict = {}
                for key, val in content.items():
                    if isinstance(val, dict) and "value" in val:
                        if isinstance(val["value"], list):
                            text_value = " ".join(map(str, val["value"]))
                        else:
                            text_value = val["value"]
                        new_content_dict[key] = text_value
                    else:
                        new_content_dict[key] = val

                review_contents.append(
                    ' '.join([new_content_dict[key] for key in keys_to_extract if key in new_content_dict]))

        return review_contents

    def _segment_content(self, content: List[str]) -> List[str]:
        """
        Segment the content of the reviews provided into single sentences.
        :param content: The review content
        :return: List of sentences.
        """

        combined_text = "".join(content)

        # I don't understand anything but regex is love <3
        list_marker_pattern = re.compile(r'''
            (^|\n)                  # Start of line or newline
            \s*                     # Any whitespace characters
            (
                (\d+(\.\d+)*[\.\)\-]?)   # Matches numbers like '1', '2.1', '3.2.1', followed by '.', ')', or '-'
                |                     # OR
                [\-\*\•\●\•]          # Matches bullet points like '-', '*', '•', '●'
            )
            \s*                      # Any whitespace
        ''', re.VERBOSE)

        # Remove the list markers from the text
        clean_text = re.sub(list_marker_pattern, '', combined_text)

        doc = self.nlp(clean_text)
        sentences = [sent.text.strip() for sent in doc.sents]

        standalone_marker_pattern = re.compile(r'^(\d+(\.\d+)*[\.\)\-]?|[\-\*\•\●\•])$')

        # Filter out sentences that are just list markers
        final_sentences = [s for s in sentences if not standalone_marker_pattern.match(s)]

        return final_sentences

    def create_testset(self, ids: List[str]) -> List[str]:
        """
        Create a test set from multiple paper reviews.
        :param ids: The ids of the papers.
        :return: List with reviews segmented in sentences.
        """

        reviews = self.get_reviews_from_multiple_papers(ids)
        processed_reviews = self._process_reviews(reviews)
        test_set = self._segment_content(processed_reviews)

        return test_set
