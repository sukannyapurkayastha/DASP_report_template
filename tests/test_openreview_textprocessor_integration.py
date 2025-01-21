import pandas as pd
import pandas.testing as pdt

from backend.text_processing.text_processor import TextProcessor
from frontend.clients.openreviewclient import OpenReviewClient


def test_openreview_textprocessing_integration(username, password):
    """
    This integration tests fetches reviews from OpenReview and processes them with TextProcessor.
    """

    client = OpenReviewClient(username, password)

    paper_id = "aVh9KRZdRk"  # openreview.net/forum?id=aVh9KRZdRk
    paper = client.get_reviews_from_id(paper_id)

    reviews = paper.reviews

    assert paper is not None
    assert reviews

    processor = TextProcessor(reviews)
    df_sentences, df_overview = processor.process()

    assert not df_sentences.empty
    assert not df_overview.empty
    assert df_sentences is not None
    assert df_overview is not None

    expected_overview_cols = ["Category", "Avg_Score", "Individual_scores"]

    assert df_overview.columns.tolist() == expected_overview_cols

    expected_first_sentence = ("The paper studies the emergence of the in-context ability of the GPT-style transformer "
                               "model trained using autoregressive loss and arithmetic modular datasets. It analyzes "
                               "the influence of the number of tasks, number of in-context examples, model capacity, "
                               "etc., on the ICL capability of an appropriately trained model (i.e., using early "
                               "stopping). It also provides a persuasive “task decomposition hypothesis”, which is "
                               "well supported by the ablation study and various experiments. The white-box analysis "
                               "on the attention heads provides convincing evidence of the proposed explanation. "
                               "Although there is a gap between the grokking settings (i.e., small model and toy "
                               "dataset) and practical systems, the paper does a good job of explaining many "
                               "important trends and concepts related to the emergence of compositional "
                               "in-context ability. I enjoy reading this paper and suggest an acceptance.")

    assert expected_first_sentence == df_sentences.iloc[0, 2]

    df_overview_expected = pd.DataFrame({
        "Category": ["Rating", "Soundness", "Presentation", "Contribution"],
        "Avg_Score": [7.25, 3.25, 3.50, 3.25],
        "Individual_scores": [
            [
                ["Reviewer rwBm", 7.0],
                ["Reviewer PjEW", 8.0],
                ["Reviewer Jerg", 7.0],
                ["Reviewer CrUb", 7.0],
            ],
            [
                ["Reviewer rwBm", 3.0],
                ["Reviewer PjEW", 4.0],
                ["Reviewer Jerg", 3.0],
                ["Reviewer CrUb", 3.0],
            ],
            [
                ["Reviewer rwBm", 4.0],
                ["Reviewer PjEW", 4.0],
                ["Reviewer Jerg", 3.0],
                ["Reviewer CrUb", 3.0],
            ],
            [
                ["Reviewer rwBm", 3.0],
                ["Reviewer PjEW", 4.0],
                ["Reviewer Jerg", 3.0],
                ["Reviewer CrUb", 3.0],
            ],
        ]
    })

    pdt.assert_frame_equal(df_overview, df_overview_expected, check_like=True)

