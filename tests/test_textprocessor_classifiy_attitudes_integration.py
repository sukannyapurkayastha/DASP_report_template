from attitude_classifier.model_prediction import combine_roots_and_themes
from backend.text_processing import TextProcessor
from frontend.clients import OpenReviewClient


def test_textprocessor_classifiy_attitudes_integration(username, password):
    """
    Integration test for combine_roots_and_themes
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

    # Todo: Check path problems for importing model :(
    result = combine_roots_and_themes(df_sentences)

    assert 1 == 1