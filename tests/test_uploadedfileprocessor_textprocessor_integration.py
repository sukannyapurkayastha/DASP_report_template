import pytest
from pathlib import Path
import pandas as pd
import pandas.testing as pdt

from backend.text_processing.text_processor import TextProcessor
from frontend.clients.models import Review
from frontend.clients.uploaded_file_processor import UploadedFileProcessor
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.proto.Common_pb2 import FileURLs as FileURLsProto


@pytest.fixture
def sample_uploaded_review():
    """
    Loads a sample .docm file from 'tests/resources/sample_review_form.docm' and returns it as a Streamlit UploadedFile
    object.
    """
    try:
        docm_path = Path("tests/resources/sample_review.docm")
        file_bytes = open(docm_path, "rb").read()
    except FileNotFoundError:
        # if started with PyCharm debugger the project path is already one deeper
        docm_path = Path("resources/sample_review.docm")
        file_bytes = open(docm_path, "rb").read()

    file_record = UploadedFileRec(file_id="2c6cff9b-8266-4d91-9353-a2c1d5a8e2a8",
                                  name="sample_review.docm",
                                  type="application/vnd.ms-word.document.macroEnabled.12",
                                  data=file_bytes
                                  )

    file_urls = FileURLsProto(file_id="2c6cff9b-8266-4d91-9353-a2c1d5a8e2a8",
                              upload_url="/_stcore/upload_file/c075ce9f-812e-4c50-82ac-afac73bb26d6/2c6cff9b-8266-4d91-9353-a2c1d5a8e2a8",
                              delete_url="/_stcore/upload_file/c075ce9f-812e-4c50-82ac-afac73bb26d6/2c6cff9b-8266-4d91-9353-a2c1d5a8e2a8"
                              )

    uploaded_file = UploadedFile(file_record, file_urls)
    return uploaded_file


def test_uploadedfileprocessor_textprocessor_integration(sample_uploaded_review):
    """
    This integration tests mocks a streamlit uploaded file and passes it to the UploadedFileProcessor and extracts
    review objects from it. The reviews then get passet to the TextProcessor.
    """

    processor = UploadedFileProcessor(uploaded_files=[sample_uploaded_review])

    reviews = processor.process()

    assert isinstance(reviews, list)
    assert len(reviews) == 1
    for r in reviews:
        assert isinstance(r, Review)
        # Check that form fields are populated
        assert hasattr(r, "summary")

    processor = TextProcessor(reviews)
    df_sentences, df_overview = processor.process()

    assert not df_sentences.empty
    assert not df_overview.empty

    expected_overview_cols = ["Category", "Avg_Score", "Individual_scores"]
    assert df_overview.columns.tolist() == expected_overview_cols

    expected_first_sentence = ("This paper presents a method of learning dense 3D correspondence between shapes in a "
                               "self-supervised manner. Specifically, it is built on an existing SO(3)-equivariant "
                               "representation. The input point clouds are independently encoded to SO(3)-equivariant "
                               "global shape descriptor Z and dynamic SO(3)-invariant point-wise local shape "
                               "transforms. Then the network is trained via penalizing errors in self- and cross- "
                               "reconstructions via the decoder. The experiment validates the effectiveness of "
                               "the proposed method.")

    assert expected_first_sentence == df_sentences.iloc[0, 2]

    df_overview_expected = pd.DataFrame({
        "Category": ["Rating", "Soundness", "Presentation", "Contribution"],
        "Avg_Score": [5.0, 3.0, 3.0, 2.0],
        "Individual_scores": [
            [["Reviewer 1", 5.0]],
            [["Reviewer 1", 3.0]],
            [["Reviewer 1", 3.0]],
            [["Reviewer 1", 2.0]]
        ]
    })

    pdt.assert_frame_equal(df_overview, df_overview_expected, check_like=True)
