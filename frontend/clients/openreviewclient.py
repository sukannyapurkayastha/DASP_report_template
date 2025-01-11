import openreview
from loguru import logger

from frontend.clients.models import Paper, Review


class OpenReviewClient:
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

    def _get_paper(self, id: str) -> Paper:
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

    def _prepare_reviews(self, notes: list[dict]) -> list[Review]:
        """
        Prepares a list reviews.
        :param notes: The openreview Notes as a list of dict to prepare.
        :return: A list of prepared reviews.
        """

        prepared_reviews = []
        for note in notes:
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
            prepared_reviews.append(review)

        return prepared_reviews

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
            prepared_reviews = self._prepare_reviews(replies)
            return prepared_reviews
        except Exception as e:
            logger.error("Failed to fetch paper reviews from OpenReview.")
            logger.error(e)
            return []

    def get_reviews_from_id(self, id) -> Paper:
        """
        Load all reviews from a specific paper.
        :param id: The id of the paper.
        :return: All reviews for a paper.
        """

        paper = self._get_paper(id)
        reviews = self._get_reviews(paper)

        paper.reviews = reviews

        return paper
