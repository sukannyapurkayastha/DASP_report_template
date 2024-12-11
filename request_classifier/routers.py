from typing import List

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

from classification.request_classifier_pipeline import process_dataframe_request
from util.utils import download_repo_files

router = APIRouter()


class RawInput(BaseModel):
    data: list[dict]


@router.post("/classify_request")
async def classify_request(request: RawInput) -> list[dict]:
    try:
        data = request.data
        df_sentences = pd.DataFrame(data)
    except Exception:
        df_sentences = pd.read_csv("sentences_author.csv")

    # Check if model exists for request classifier
    repo_id = "JohannesLemken/DASP_models"
    subdir = "Request_Classifier/RequestClassifier/"
    local_dir_request_classifier = "models/request_classifier/"
    local_dir_fine_request_classifier = "models/fine_request_classifier/"

    # not needed actually
    # download_repo_files(repo_id=repo_id, subdir=subdir, local_dir=local_dir_request_classifier)

    df_result = process_dataframe_request(df=df_sentences, local_dir=local_dir_request_classifier,
                                          local_dir_fine_request=local_dir_fine_request_classifier)

    df_result_dict = df_result.to_dict(orient="records")

    return df_result_dict
