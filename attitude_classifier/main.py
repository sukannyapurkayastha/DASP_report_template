from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn
from model_prediction import (
    combine_roots_and_themes
)
from loguru import logger

app = FastAPI()

# Define schema for incoming preprocessed data
class RawInput(BaseModel):
    data: list[dict]

@app.post("/classify_attitudes")
async def predict(request: RawInput):
    """
    Endpoint for classifying attitudes based on input data.

    This function receives a request containing sentence data, processes it using 
    the `combine_roots_and_themes` function, and returns a structured classification 
    result.

    Args:
        request (RawInput): The input request containing a `data` attribute with 
                            sentence data in JSON format.

    Returns:
        list[dict]: A list of dictionaries where each dictionary represents a 
                    processed sentence with classification results.

    Raises:
        HTTPException: If an error occurs during processing, a 500 error is returned
                       with the error details.
    """
    try:
        data = request.data
        df_sentences = pd.DataFrame(data)

        logger.info("Predicting attitudes...")
        result = combine_roots_and_themes(df_sentences)
        # TODO: description
        result = result.fillna('none')
        # Convert dataframes to JSON for response
        result = result.to_dict(orient="records")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
    
# Run the application on port 8082
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)

