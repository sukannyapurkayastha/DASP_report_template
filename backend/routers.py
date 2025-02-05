import pandas as pd
import requests
from fastapi import APIRouter, HTTPException
from data_processing import process_file
from loguru import logger
from pydantic import BaseModel
import os

router = APIRouter()


class RawInput(BaseModel):
    data: list[dict]  # Raw input data in JSON format


@router.post("/process")
async def process_data(input_data: RawInput) -> dict:
    try:
        # Step 1: Process raw data
        df_overview, df_sentences = process_file(input_data.data)
        overview = df_overview.to_dict(orient='records')
        sentences = df_sentences.to_dict(orient='records')
        logger.success("Got overview and sentences df.")


        # Run attitude classifer
        attitude_classifier_url = os.environ.get("ATTITUDE_CLASSIFIER_URL", "http://localhost:8082")
        try:
            logger.info(f"Tyring to post: {'http://attitude_classifier:8082/classify_attitudes'}")

            response_attitude = requests.post(
                f"{attitude_classifier_url}/classify_attitudes",
                json={"data": sentences}
            )
            if response_attitude.status_code == 200:
                attitude_data = response_attitude.json()
                logger.success("Attitude classification successful.")

            else:
                logger.error(f"Model Attitude Classifier API Error: {response_attitude.text}")
        except Exception as e:
            logger.error(f"Error communicating with Model Attitude Classifier: {e}")

        # Run request classifier
        request_classifier_url = os.environ.get("REQUEST_CLASSIFIER_URL", "http://localhost:8081")
        try:
            logger.info(f"Tyring to post: {'http://request_classifier:8081/classify_request'}")
            response_request = requests.post(
                f"{request_classifier_url}/classify_request",
                json={"data": sentences}
            )

            if response_request.status_code == 200:
                request_data = response_request.json()
                logger.success("Request classification successful.")
                # request_response = pd.DataFrame(data)

            else:
                logger.error(f"Model Request Classifier API Error: {response_request.text}")
        except Exception as e:
            logger.error(f"Error communicating with Model Request Classifier: {e}")
        
        
        # Run summary generator
        summary_generator_url = os.environ.get("SUMMARY_GENERATOR_URL", "http://localhost:8083")
        try: 
            logger.info(f"Tyring to post: {'http://summary_generator:8083/generate_summary'}")
            response_summary = requests.post(
                f"{summary_generator_url}/generate_summary",
                json={"overview_df": {"data": overview}, 
                      "attitude_df": {"data": pd.DataFrame(attitude_data).to_dict(orient='records')}, 
                      "request_df": {"data": pd.DataFrame(request_data).to_dict(orient='records')}}
            )
            if response_summary.status_code == 200:
                summary_data = response_summary.json()
                logger.success("Summary generation successful.")
            else:
                logger.error(f"Model Summary Generator API Error: {response_summary.text}")
        except Exception as e:
            logger.error(f"Error communicating with model Summary Generator: {e}")
        
        
        return {
            "overview": overview,
            "request_response": request_data,
            "attitude_response": attitude_data,
            "summary_response": summary_data
        }
        
        
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
