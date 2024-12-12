import pandas as pd
import pytest

from backend.text_processing import Review, TextProcessor


@pytest.fixture
def mock_reviews():
    # Provide a list of mock Review objects
    return [
        Review(
            reviewer="Reviewer 1",
            summary="This is a summary.\nIt has two lines.",
            strengths="Strong points:\n* Clarity\n* Relevance",
            weaknesses="Weak points:\n* Formatting\n* Length",
            questions="Could you clarify?\nWhat is the main contribution?",
            rating="5: marginally below the acceptance threshold",
            soundness="3 good",
            presentation="3 good",
            contribution="2 fair",
            confidence="4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work."
        ),
        Review(
            reviewer="Reviewer 2",
            summary="Another summary.\nWith multiple lines.",
            strengths="Strength:\n* Methodology\n* Results",
            weaknesses="Weakness:\n* Limited examples",
            questions="Any future directions?\nWhat about scalability?",
            rating="6: marginally above the acceptance threshold",
            soundness="2 fair",
            presentation="2 fair",
            contribution="2 fair",
            confidence="2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked."
        )
    ]


def test_preprocess_text(mock_reviews):
    processor = TextProcessor(mock_reviews)
    preprocessed = processor._preprocess_text(mock_reviews)

    for review in preprocessed:
        # Check that the text is stripped and line breaks are handled
        assert not any(line.endswith(" ") for line in review.summary.split("\n"))
        # Ensure that no HTTP links remain
        assert "http" not in review.summary.lower()
        # Check that semicolons and asterisks were replaced/removed
        assert ";" not in review.summary
        assert "*" not in review.summary


def test_segment_content(mock_reviews):
    processor = TextProcessor(mock_reviews)
    preprocessed = processor._preprocess_text(mock_reviews)
    reviews, df_sentences = processor._segment_content(preprocessed)

    # Check if df_sentences is a DataFrame
    assert isinstance(df_sentences, pd.DataFrame)

    # Check required columns
    assert "author" in df_sentences.columns
    assert "tag" in df_sentences.columns
    assert "sentence" in df_sentences.columns

    # Ensure DataFrame is not empty
    assert not df_sentences.empty

    # Check that each processed review now has a `sentences` attribute
    for review in reviews:
        assert hasattr(review, 'sentences')
        assert isinstance(review.sentences, list)
        # There should be at least one sentence after segmentation
        assert len(review.sentences) > 0


def test_get_overview(mock_reviews):
    processor = TextProcessor(mock_reviews)
    preprocessed = processor._preprocess_text(mock_reviews)
    df_overview = processor._get_overview(preprocessed)

    # Check the structure of the overview DataFrame
    assert isinstance(df_overview, pd.DataFrame)
    for col in ["Category", "Avg_Score", "Individual_scores"]:
        assert col in df_overview.columns

    # Check the correctness of averages:
    # For rating: Reviewer 1 has a score of 4, Reviewer 2 has a score of 3, so average should be 3.5
    rating_row = df_overview[df_overview["Category"] == "Rating"]
    assert not rating_row.empty
    avg_rating = rating_row["Avg_Score"].values[0]
    assert avg_rating == pytest.approx(5.5, 0.001)


def test_process_method(mock_reviews):
    processor = TextProcessor(mock_reviews)
    df_sentences, df_overview = processor.process()

    # Check returned DataFrames
    assert isinstance(df_sentences, pd.DataFrame)
    assert isinstance(df_overview, pd.DataFrame)

    # Check that sentence segmentation happened
    assert not df_sentences.empty
    assert "sentence" in df_sentences.columns

    # Check overview DataFrame
    assert not df_overview.empty
    for col in ["Category", "Avg_Score", "Individual_scores"]:
        assert col in df_overview.columns

    # Spot check a known average
    rating_row = df_overview[df_overview["Category"] == "Rating"]
    assert not rating_row.empty
    avg_rating = rating_row["Avg_Score"].values[0]
    # With the given mock data, the average rating should still be 3.5 (4 and 3)
    assert avg_rating == pytest.approx(5.5, 0.001)
