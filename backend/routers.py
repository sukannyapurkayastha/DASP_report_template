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

        # Step 2: Run predictions
        try:
            response = requests.post(
                "http://localhost:8081/classify_request",
                json={"data": sentences}
            )

            if response.status_code == 200:
                request_data = response.json()
                # request_response = pd.DataFrame(data)

            else:
                logger.error(f"API Error: {response.text}")
        except Exception as e:
            logger.error(f"Error: {e}")

        # Todo: Call api to other containers here
        # dataframe_a = combine_roots_and_themes(processed_data)
        # dataframe_1 = model_prediction_1(processed_data)
        # dataframe_2 = model_prediction_2(processed_data)
        # dataframe_3 = model_prediction_3(processed_data)
        # dataframe_4 = model_prediction_4(processed_data)

        # # Step 3: Combine predictions into dataframes for charts
        # dataframe_a = combine_for_a(dataframe_1, dataframe_2)
        # dataframe_b = combine_for_b(dataframe_3, dataframe_4)

        # Step 4: Convert dataframes to JSON for response
        # result = dataframe_a.to_dict(orient="records")

        return {
            "overview": overview,
            "request_response": request_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
