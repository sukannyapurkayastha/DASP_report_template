import pytest
from unittest.mock import patch, MagicMock

from openreview.api.client import Note

from frontend.clients.openreviewclient import OpenReviewClient
from frontend.clients.models import Paper, Review


@pytest.fixture
def mock_openreview_client():
    """
    Fixture to mock openreview.api.OpenReviewClient so that we do not make api calls in tests.
    """
    with patch("openreview.api.OpenReviewClient") as mock_client_class:
        mock_client_instance = mock_client_class.return_value
        yield mock_client_instance


def test_is_review_true():
    """
    Test _is_review method when 'Official_Review' is present in invitations.
    """
    invitations = ["Conference/2024/Paper1/Official_Review", "Conference/2024/Paper1/Meta_Review"]
    assert OpenReviewClient._is_review(invitations) is True


def test_is_review_false():
    """
    Test _is_review method when 'Official_Review' is NOT present in invitations.
    """
    invitations = ["Conference/2024/Paper1/Meta_Review"]
    assert OpenReviewClient._is_review(invitations) is False


def test_is_paper_true():
    """
    Test _is_paper method when note has 'title' and 'abstract' in content.
    """
    mock_note = MagicMock()
    mock_note.content = {
        "title": {"value": "A Great Paper"},
        "abstract": {"value": "This is the abstract."},
        "other_field": {"value": "Some other data"}
    }
    assert OpenReviewClient._is_paper(mock_note) is True


def test_is_paper_false():
    """
    Test _is_paper method when note is missing either 'title' or 'abstract' in content.
    """
    mock_note = MagicMock()
    mock_note.content = {
        "title": {"value": "A Great Paper"},
        # 'abstract' is missing
        "other_field": {"value": "Some other data"}
    }
    assert OpenReviewClient._is_paper(mock_note) is False


def test_get_paper_success(mock_openreview_client):
    """
    Test _get_paper method on success scenario with a properly returned paper note.
    """
    # Excerpt from the paper https://openreview.net/forum?id=aVh9KRZdRk
    mock_openreview_client.get_all_notes.return_value = [Note(
        id="aVh9KRZdRk",
        forum="aVh9KRZdRk",
        domain='NeurIPS.cc/2024/Conference',
        replyto=None,
        invitations=[
            "NeurIPS.cc/2024/Conference/-/Submission",
            "NeurIPS.cc/2024/Conference/-/Post_Submission",
            "NeurIPS.cc/2024/Conference/Submission21497/-/Revision",
            "NeurIPS.cc/2024/Conference/-/Edit",
            "NeurIPS.cc/2024/Conference/Submission21497/-/Camera_Ready_Revision",
        ],
        readers=["everyone"],
        writers=["NeurIPS.cc/2024/Conference", "NeurIPS.cc/2024/Conference/Submission21497/Authors"],
        signatures=["NeurIPS.cc/2024/Conference/Submission21497/Authors"],
        content={
            "title": {
                "value": "Learning to grok: Emergence of in-context learning and skill composition in modular arithmetic tasks"
            },
            "abstract": {
                "value": "Large language models can solve tasks that were not present in the training set. ... (truncated)"
            },
            "keywords": {"value": ["In-Context Learning", "Grokking", "Modular Arithmetic", "Interpretability"]},
            "venue": {"value": "NeurIPS 2024 oral"},
            "primary_area": {"value": "interpretability_and_explainability"},
        },
        details={
            "directReplies": [
                {
                    "id": "PdJabwbJeE",
                    "forum": "aVh9KRZdRk",
                    "content": {
                        "summary": {"value": "This is a sample direct reply summary."},
                        "rating": {"value": 7},
                    },
                    "invitations": [
                        "NeurIPS.cc/2024/Conference/Submission21497/-/Official_Review"
                    ],
                }
            ],
        },
        cdate=1715802154333,
        mdate=1730874005890,
        tcdate=1715802154333,
        tmdate=1730874005890,
    )
    ]

    client = OpenReviewClient(username="test_user", password="test_pass")
    paper = client._get_paper("aVh9KRZdRk")

    assert paper is not None
    assert isinstance(paper, Paper)

    assert paper.id == "aVh9KRZdRk"
    assert paper.title == (
        "Learning to grok: Emergence of in-context learning "
        "and skill composition in modular arithmetic tasks"
    )
    assert paper.keywords == [
        "In-Context Learning",
        "Grokking",
        "Modular Arithmetic",
        "Interpretability",
    ]
    assert "Large language models can solve tasks" in paper.abstract
    assert paper.primary_area == "interpretability_and_explainability"
    assert paper.venue == "NeurIPS 2024 oral"
    assert paper.cdate == 1715802154333
    assert paper.domain == "NeurIPS.cc/2024/Conference"
    assert paper.forum == "aVh9KRZdRk"

    # directReplies should match what the mock returned
    assert len(paper.directReplies) == 1
    assert paper.directReplies[0]["id"] == "PdJabwbJeE"
    assert "summary" in paper.directReplies[0]["content"]


def test_get_paper_failure(mock_openreview_client):
    """
    Test _get_paper when no valid paper note is found.
    """
    mock_openreview_client.get_all_notes.return_value = [
        Note(
            id="some_id",
            content={
                "title": {"value": "But I'm missing an abstract"},
                # No 'abstract'
            }
        )
    ]

    client = OpenReviewClient(username="test_user", password="test_pass")
    paper = client._get_paper("invalid_paper_id")

    # We expect None due to the note not being a valid paper
    assert paper is None


def test_prepare_reviews(mock_openreview_client):
    """
    Test the _prepare_reviews method by providing mock notes (in the form of Note objects).
    """
    notes = [
        {
            "id": "PdJabwbJeE",
            "forum": "aVh9KRZdRk",
            "replyto": "aVh9KRZdRk",
            "signatures": ["NeurIPS.cc/2024/Conference/Submission21497/Reviewer_rwBm"],
            "nonreaders": [],
            "readers": ["everyone"],
            "writers": ["NeurIPS.cc/2024/Conference", "NeurIPS.cc/2024/Conference/Submission21497/Reviewer_rwBm"],
            "number": 1,
            "invitations": [
                "NeurIPS.cc/2024/Conference/Submission21497/-/Official_Review",
                "NeurIPS.cc/2024/Conference/-/Edit"
            ],
            "domain": "NeurIPS.cc/2024/Conference",
            "tcdate": 1718340492945,
            "cdate": 1718340492945,
            "tmdate": 1730880178832,
            "mdate": 1730880178832,
            "license": "CC BY 4.0",
            "version": 2,
            "content": {
                "summary": {
                    "value": (
                        "The paper studies the emergence of the in-context "
                        "ability of the GPT-style transformer model..."
                    )
                },
                "soundness": {"value": 3},
                "presentation": {"value": 4},
                "contribution": {"value": 3},
                "strengths": {
                    "value": (
                        "- The paper is easy to follow. Good presentation!\n"
                        "- The experiments are well-designed, providing compelling support for the claims.\n"
                        "    - The results in Figure 5 are cool.\n"
                        "    - The skill decomposition discussed in section 5 is great. The clear pattern in "
                        "attention heads verifies it very well. (The hypotheses could be further verified...)"
                    )
                },
                "weaknesses": {
                    "value": (
                        "- The emergent ability (or grokking) usually refers to a phenomenon in the model “got stuck” "
                        "in a non-generalization region and suddenly gained the generalization ability..."
                    )
                },
                "questions": {
                    "value": (
                        "- The paper claims in line 147 that 'As the o.o.d. performance increases, "
                        "the pre-training performance simultaneously degrades'. However, it is hard to read..."
                    )
                },
                "limitations": {"value": "Discussions on how the findings help the practical system."},
                "flag_for_ethics_review": {"value": ["No ethics review needed."]},
                "rating": {"value": 7},
                "confidence": {"value": 4},
                "code_of_conduct": {"value": "Yes"}
            }
        }
    ]

    client = OpenReviewClient(username="test", password="test")

    prepared_reviews = client._prepare_reviews(notes)

    assert len(prepared_reviews) == 1
    review = prepared_reviews[0]
    assert isinstance(review, Review)
    assert review.id == "PdJabwbJeE"
    assert review.forum == "aVh9KRZdRk"
    assert review.venue == "NeurIPS.cc"
    assert review.year == "2024"
    assert review.type == "Conference"
    assert review.reviewer == "Reviewer rwBm"
    assert review.date == 1718340492945
    assert review.summary.startswith("The paper studies the emergence")
    assert review.rating == 7
    assert review.confidence == 4
    assert review.weaknesses.startswith("- The emergent ability (or grokking)")


def test_get_reviews(mock_openreview_client):
    """
    Test that _get_reviews filters out the correct replies (official reviews) and returns them.
    """
    paper = Paper(
        title=(
            "Learning to grok: Emergence of in-context learning and skill composition in modular arithmetic tasks"
        ),
        keywords=["In-Context Learning", "Grokking", "Modular Arithmetic", "Interpretability"],
        abstract=(
            "Large language models can solve tasks that were not present in the training set. This capability is "
            "believed to be due to in-context learning and skill composition. ..."
        ),
        primary_area="interpretability_and_explainability",
        venue="NeurIPS 2024 oral",
        cdate=1715802154333,
        domain="NeurIPS.cc/2024/Conference",
        forum="aVh9KRZdRk",
        id="aVh9KRZdRk",
        directReplies=[
            {
                "id": "PdJabwbJeE",
                "invitations": ["NeurIPS.cc/2024/Conference/Submission21497/-/Official_Review"],
                "forum": "aVh9KRZdRk",
                "domain": "NeurIPS.cc/2024/Conference",
                "cdate": 1718340492945,
                "content": {
                    "summary": {
                        "value": (
                            "The paper studies the in-context ability of the GPT-style "
                            "transformer model trained on modular arithmetic tasks..."
                        )
                    },
                    "strengths": {
                        "value": "- The paper is easy to follow. Good presentation!\n- The experiments are well-designed, providing compelling support for the claims.\n"},
                    "weaknesses": {
                        "value": "- The emergent ability (or grokking) usually refers to a phenomenon in the model “got stuck” in a non-generalization region and suddenly gained the generalization ability"},
                    "questions": {
                        "value": '- The paper claims in line 147 that “As the o.o.d. performance increases, the pre-training performance simultaneously degrades “. However, it is hard to read this information from Figure 3-a panel 1. Maybe a different color mapping or adding numbers on these patches would be helpful.'},
                    "soundness": {"value": 3},
                    "presentation": {"value": 4},
                    "contribution": {"value": 3},
                    "rating": {"value": 7},
                    "confidence": {"value": 4},
                    "flag_for_ethics_review": {"value": ["No ethics review needed."]},
                },
                "writers": ["NeurIPS.cc/2024/Conference/Submission21497/Reviewer_rwBm"],
            },
            {
                "id": "comment1",
                "invitations": ["NeurIPS.cc/2024/Conference/Submission21497/-/Comment"],
                "forum": "aVh9KRZdRk",
                "domain": "NeurIPS.cc/2024/Conference/Comment",
                "content": {
                    "comment": {"value": "This is a random comment, not an official review."}
                },
                "writers": ["NeurIPS.cc/2024/Conference/Submission21497/Commenter"],
            },
        ],
    )

    client = OpenReviewClient(username="test_user", password="test_pass")
    reviews = client._get_reviews(paper)

    assert len(reviews) == 1, "Only one official review should be returned."
    review = reviews[0]
    assert review.id == "PdJabwbJeE"
    assert review.forum == "aVh9KRZdRk"
    assert review.type == "Conference"
    assert review.venue == "NeurIPS.cc"
    assert review.year == "2024"
    assert review.rating == 7
    assert review.confidence == 4
    assert review.reviewer == "Reviewer rwBm"


def test_get_reviews_from_id(mock_openreview_client):
    """
    End-to-end test for get_reviews_from_id. It should fetch the paper, then gather reviews, returning a Paper
    object with `reviews` populated.
    """
    mock_openreview_client.get_all_notes.return_value = [Note(
        id="aVh9KRZdRk",
        forum="aVh9KRZdRk",
        domain="NeurIPS.cc/2024/Conference",
        replyto=None,
        invitations=[
            "NeurIPS.cc/2024/Conference/-/Submission",
            "NeurIPS.cc/2024/Conference/-/Post_Submission",
            "NeurIPS.cc/2024/Conference/Submission21497/-/Revision",
            "NeurIPS.cc/2024/Conference/-/Edit",
            "NeurIPS.cc/2024/Conference/Submission21497/-/Camera_Ready_Revision",
        ],
        readers=["everyone"],
        writers=["NeurIPS.cc/2024/Conference", "NeurIPS.cc/2024/Conference/Submission21497/Authors"],
        signatures=["NeurIPS.cc/2024/Conference/Submission21497/Authors"],
        content={
            "title": {
                "value": (
                    "Learning to grok: Emergence of in-context learning and "
                    "skill composition in modular arithmetic tasks"
                )
            },
            "abstract": {
                "value": (
                    "Large language models can solve tasks that were not present in the training set. "
                    "This capability is believed to be due to in-context learning and skill composition. "
                    "... (truncated)"
                )
            },
            "keywords": {"value": ["In-Context Learning", "Grokking", "Modular Arithmetic", "Interpretability"]},
            "venue": {"value": "NeurIPS 2024 oral"},
            "primary_area": {"value": "interpretability_and_explainability"},
        },
        details={
            "directReplies": [
                {
                    "id": "PdJabwbJeE",
                    "invitations": ["NeurIPS.cc/2024/Conference/Submission21497/-/Official_Review"],
                    "forum": "aVh9KRZdRk",
                    "domain": "NeurIPS.cc/2024/Conference",
                    "cdate": 1718340492945,
                    "content": {
                        "summary": {
                            "value": (
                                "The paper studies the in-context ability of the GPT-style "
                                "transformer model trained on modular arithmetic tasks..."
                            )
                        },
                        "strengths": {
                            "value": "- The paper is easy to follow. Good presentation!\n- The experiments are well-designed, providing compelling support for the claims.\n"},
                        "weaknesses": {
                            "value": "- The emergent ability (or grokking) usually refers to a phenomenon in the model “got stuck” in a non-generalization region and suddenly gained the generalization ability"},
                        "questions": {
                            "value": '- The paper claims in line 147 that “As the o.o.d. performance increases, the pre-training performance simultaneously degrades “. However, it is hard to read this information from Figure 3-a panel 1. Maybe a different color mapping or adding numbers on these patches would be helpful.'},
                        "soundness": {"value": 3},
                        "presentation": {"value": 4},
                        "contribution": {"value": 3},
                        "rating": {"value": 7},
                        "confidence": {"value": 4},
                        "flag_for_ethics_review": {"value": ["No ethics review needed."]},
                    },
                    "writers": ["NeurIPS.cc/2024/Conference/Submission21497/Reviewer_rwBm"],
                },
                {
                    "id": "comment1",
                    "forum": "aVh9KRZdRk",
                    "invitations": [
                        "NeurIPS.cc/2024/Conference/Submission21497/-/Comment"
                    ],
                    "content": {
                        "comment": {"value": "This is a random comment, not an official review."}
                    },
                    "writers": [
                        "NeurIPS.cc/2024/Conference/Submission21497/Commenter"
                    ],
                },
            ],
        },
        cdate=1715802154333,
        mdate=1730874005890,
        tcdate=1715802154333,
        tmdate=1730874005890,
    )
    ]

    client = OpenReviewClient(username="test_user", password="test_pass")
    paper = client.get_reviews_from_id("aVh9KRZdRk")

    assert paper is not None
    assert isinstance(paper, Paper)
    assert paper.id == "aVh9KRZdRk"
    assert paper.title.startswith("Learning to grok:")
    assert "Large language models can solve tasks" in paper.abstract
    assert paper.keywords == ["In-Context Learning", "Grokking", "Modular Arithmetic", "Interpretability"]
    assert paper.primary_area == "interpretability_and_explainability"
    assert paper.venue == "NeurIPS 2024 oral"
    assert paper.cdate == 1715802154333
    assert paper.domain == "NeurIPS.cc/2024/Conference"
    assert paper.forum == "aVh9KRZdRk"

    assert paper.reviews is not None
    assert len(paper.reviews) == 1
    review = paper.reviews[0]
    assert isinstance(review, Review)

    assert review.id == "PdJabwbJeE"
    assert review.forum == "aVh9KRZdRk"
    assert review.type == "Conference"
    assert review.venue == "NeurIPS.cc"
    assert review.year == "2024"
    assert review.reviewer == "Reviewer rwBm"
    assert review.rating == 7
    assert review.confidence == 4
    assert review.date == 1718340492945
