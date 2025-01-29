from backend.text_processing import TextProcessor
from frontend.clients import OpenReviewClient
import requests
import pandas as pd



def test_textprocessor_classifiy_attitudes_integration(username, password):
    """
    Integration test for combine_roots_and_themes
    """
    openreview_client = OpenReviewClient(username, password)

    paper_id = "aVh9KRZdRk"  # openreview.net/forum?id=aVh9KRZdRk
    paper = openreview_client.get_reviews_from_id(paper_id)

    reviews = paper.reviews

    assert paper is not None
    assert reviews

    processor = TextProcessor(reviews)
    df_sentences, df_overview = processor.process()

    assert not df_sentences.empty
    assert not df_overview.empty
    assert df_sentences is not None
    assert df_overview is not None

    sentences = df_sentences.to_dict(orient='records')
    response = requests.post("http://localhost:8082/classify_attitudes", json={"data": sentences})

    assert response.status_code == 200

    df = pd.DataFrame(response.json())

    assert df is not None
    assert df.shape == (27, 4)
    assert df.columns.tolist() == ['Attitude_roots', 'Frequency', 'Descriptions', 'Comments']
