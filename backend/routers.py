from fastapi import APIRouter, HTTPException
from data_processing import process_file
from model_prediction import (
    combine_roots_and_themes
)
from pydantic import BaseModel

router = APIRouter()


class RawInput(BaseModel):
    data: list[dict]  # Raw input data in JSON format


@router.post("/process")
async def process_data(input_data: RawInput):
    try:
        # Step 1: Process raw data
        overview, processed_data = process_file(input_data.data)

        # Step 2: Run predictions
        dataframe_a = combine_roots_and_themes(processed_data)
        # dataframe_1 = model_prediction_1(processed_data)
        # dataframe_2 = model_prediction_2(processed_data)
        # dataframe_3 = model_prediction_3(processed_data)
        # dataframe_4 = model_prediction_4(processed_data)

        # # Step 3: Combine predictions into dataframes for charts
        # dataframe_a = combine_for_a(dataframe_1, dataframe_2)
        # dataframe_b = combine_for_b(dataframe_3, dataframe_4)

        # Step 4: Convert dataframes to JSON for response
        result = dataframe_a.to_dict(orient="records")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
