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
    try:
        data = request.data
        df_sentences = pd.DataFrame(data)
    # except Exception as e:
    #     logger.warning(f'Loading processed data failed, dummy data is in use: : {e}')
    #     df_sentences = pd.read_csv("sentences_author.csv")
        # repo_id = "JohannesLemken/DASP_models"
        # # download attitude root
        # download_repo_files(repo_id=repo_id, subdir="Attitude_Classifier/", local_dir='models/attitude_root')
        # # download attitude theme
        # download_repo_files(repo_id=repo_id, subdir="Theme_Classifier/", local_dir='models/attitude_theme')

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

