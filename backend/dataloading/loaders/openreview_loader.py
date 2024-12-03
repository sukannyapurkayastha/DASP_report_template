import openreview
import pandas as pd
from loguru import logger
from typing import Literal

from backend.dataloading.loaders.models import Paper, Review
from backend.dataloading.loaders.text_processor import TextProcessor


class OpenReviewLoader(TextProcessor):
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
        super().__init__()
        self.client = openreview.api.OpenReviewClient(
            baseurl=baseurl,
            username=username,
            password=password,
        )

        logger.info("OpenReview client initialized.")

    @staticmethod
    def _is_review(invitations: list) -> bool:
        return any("Official_Review" in invitation for invitation in invitations)

    @staticmethod
    def _is_paper(note: openreview.Note) -> bool:
        return ("title" in note.content) & ("abstract" in note.content)

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
                # authors=_paper.content["authors"]["value"],
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

    def create_testset(self, ids: list[str]) -> pd.DataFrame:
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
