import pandas as pd
import pytest

from backend.text_processing import Review, TextProcessor


@pytest.fixture
def mock_reviews():
    """
    Provide a list of mock Review objects
    """
    return [
        Review(
            reviewer="Reviewer 1",
            summary="This work proposes LSTNet, a self-supervised method to establish reliable 3D dense correspondences irrespective of the input point clouds’ rotational orientation.\n\nSpecifically, LSTNet learns to formulate SO(3)-invariant local shape transform for each point in a dynamic, input-dependent manner. Each point-wise local shape transform can map the SO(3)-equivariant global shape descriptor of the input shape to a local shape descriptor, which is passed to the decoder to reconstruct the shape and pose of the input point cloud. \n\nThe proposed self-supervised training pipeline encourages semantically corresponding points from different shape instances to be mapped to similar local shape descriptors, enabling LSTNet to establish dense point-wise correspondences via nearest point pairs between cross-reconstructed point clouds.",
            strengths="The self- and cross-reconstruction training strategy is simple yet effective. \n\nLSTNet demonstrates state-of-the-art performance on 3D semantic matching when evaluated on the KeypointNet dataset and part segmentation label transfer when evaluated on the ShapeNet dataset..",
            weaknesses="The performance of aligned shape pairs under the setting of I/I shows that other methods, such as CPAE, are much better than LSTNet.",
            questions="The reason why other methods are much better than LSTNet under the setting of I/I should be clarified.\n\nLack of limitations.?",
            rating="6 marginally above the acceptance threshold",
            soundness="3 good",
            presentation="2 fair",
            contribution="3 good",
            confidence="2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked."
        ),
        Review(
            reviewer="Reviewer 2",
            summary="1) This paper proposes a self-supervised method to find semantically corresponding points for a point cloud pair;\n\n2）The main idea is to decouple a point cloud feature learning process into a SO(3)-equivariant global shape descriptor and dynamic SO(3)-invariant point-wise local shape transforms;\n\n3) Experiments on the KeypointNet dataset show the effectiveness of the proposed method.",
            strengths="1) This paper is generally well-written;\n\n2) The idea of factorizing point cloud descriptors into SO(3)-equivariant global shape descriptor and dynamic SO(3)-invariant\npoint-wise local shape transforms seems to be novel;\n\n3) Experimental results are good.",
            weaknesses="1) The main weakness of this paper could be all experiments are performed on synthetic datasets, with simple point cloud. It's good for authors' to show some examples/experiments on real-world datasets. For example, the 3Dmatch dataset. \n\n2) Since the proposed method can estimate dense correspondences, I wonder whether the proposed method can be used to estimate the relative rotation/translation for a point cloud pair. For example, the estimated dense correspondences can be fed to an ICP method to estimate the relative rotation/translation. \n\n3) The running time and GPU memory cost is blurry for me;\n\n4) Please compare the proposed method with more recent papers, e.g., [SC3K: Self-supervised and Coherent 3D Keypoints Estimation\nfrom Rotated, Noisy, and Decimated Point Cloud Data].",
            questions="Please refer to the weaknesses.",
            rating="5: marginally below the acceptance threshold",
            soundness="3 good",
            presentation="3 good",
            contribution="2 fair",
            confidence="4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work."
        ),

    ]


def test_preprocess_text(mock_reviews):
    """
    Tests the `_preprocess_text` method of the `TextProcessor` class.

    This test verifies that:
      1. Trailing spaces are removed.
      2. Line breaks are handled properly.
      3. HTTP links are eliminated.
      4. Special characters (e.g., semicolons, asterisks) are replaced or removed.
    """
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
    """
    Tests the `_segment_content` method of the `TextProcessor` class.

    This test verifies that:
      1. A pandas DataFrame (`df_sentences`) is produced with the correct columns.
      2. Each `Review` object is augmented with a `sentences` attribute containing a list of segmented sentences.
      3. The DataFrame is not empty and contains one or more sentences for each review.
    """
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
    """
    Tests the `_get_overview` method of the `TextProcessor` class.

    This test verifies that the generated overview DataFrame has the expected
    columns and that the average scores are calculated correctly.
    """
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
    """
    Tests the `process` method of the `TextProcessor` class.

    This test verifies that:
      1. The method returns two pandas DataFrames (`df_sentences` and `df_overview`).
      2. Sentence segmentation is correctly performed.
      3. The overview DataFrame `df_overview` is not empty and contains the
         expected columns.
      4. The average rating is correctly computed.
    """
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

    rating_row = df_overview[df_overview["Category"] == "Rating"]
    assert not rating_row.empty
    avg_rating = rating_row["Avg_Score"].values[0]
    assert avg_rating == pytest.approx(5.5, 0.001)
