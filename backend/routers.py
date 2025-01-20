import pandas as pd
import requests
from fastapi import APIRouter, HTTPException
from data_processing import process_file
from loguru import logger
from pydantic import BaseModel

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

        # Run attitude classifer
        try: 
            response_attitude = requests.post(
                "http://localhost:8082/classify_attitudes",
                json={"data": sentences}
            )
            if response_attitude.status_code == 200:
                attitude_data = response_attitude.json()
            else:
                logger.error(f"Model Attitude Classifier API Error: {response_attitude.text}")
        except Exception as e:
            logger.error(f"Error communicating with Model Attitude Classifier: {e}")
            
        # Run request classifier
        try:
            response_request = requests.post(
                "http://localhost:8081/classify_request",
                json={"data": sentences}
            )

            if response_request.status_code == 200:
                request_data = response_request.json()
                # request_response = pd.DataFrame(data)

            else:
                logger.error(f"Model Request Classifier API Error: {response_request.text}")
        except Exception as e:
            logger.error(f"Error communicating with Model Request Classifier: {e}")
        
        
        # run summary generator
        """
        try: 
            response_summary = requests.post(
                "http://localhost:8083/generate_summary",
                json={"overview_df": overview, "attitude_df": attitude_data, "request_df": request_data}
            )
            if response_attitude.status_code == 200:
                summary_data = response_summary.json()
            else:
                logger.error(f"Model Summary Generator API Error: {response_summary.text}")
        except Exception as e:
            logger.error(f"Error communicating with model Summary Generator: {e}")
        
        """
        
        return {
            "overview": overview,
            "request_response": request_data,
            "attitude_response": attitude_data
            # "summary_response": summary_data
        }
        
        
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
