"""
main.py

A FastAPI application that exposes an endpoint for generating summaries
from preprocessed data.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn
from data_processing import generate_input_text, predict_data
import predict_LLAMA2
import json
from loguru import logger
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
)
import torch

app = FastAPI()

# Define schema for incoming preprocessed data
class RawInput(BaseModel):
    data: list[dict]

# Global variables for preloaded model
model = None
tokenizer = None

@app.on_event("startup")
def load_LLAMA2_model(model_dir: str = "./models/llama2"):
    """
    Load a LLaMA 2 model from Hugging Face that is supposed to be stored locally at the given path.
    For a real-world scenario, ensure you have:
      - 'transformers>=4.30'
      - 'sentencepiece'
      - You have accepted the license for LLaMA2 if it's gated.
      
      Args:
          model_dir (str): path to directory where the model is supposed to be stored.
    """
    global model, tokenizer
    logger.info(f"Loading model '{model_dir}' from Hugging Face...")

    hf_dir = "DASP-ROG/SummaryModel"
    tokenizer = LlamaTokenizer.from_pretrained(hf_dir, cache_dir=model_dir, legacy=True)
    model = LlamaForCausalLM.from_pretrained(
        hf_dir,
        torch_dtype=torch.float16,
        cache_dir=model_dir,
        device_map="auto"
    )

    # Set pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    logger.info("Model loaded successfully.")

@app.post("/generate_summary")
async def predict(overview_df: RawInput, attitude_df: RawInput, request_df: RawInput) -> list[dict]:
    """
    Generate a summary using the provided overview, attitude, and request data.
    
    Args:
        overview_df (RawInput): Overview data.
        attitude_df (RawInput): Attitude roots data.
        request_df (RawInput): Request information data.
    
    Returns:
        list[dict]: A list of summary lines formatted as dictionaries.
    """
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=500, detail="Model not loaded yet")
        
        # 1) Transform json files into dfs
        overview = overview_df.data
        overview_df = pd.DataFrame(overview)
        
        attitude = attitude_df.data
        attitude_df = pd.DataFrame(attitude)
        
        request = request_df.data
        request_df = pd.DataFrame(request)

        # 2) Transform overview into its final text and extract lists with attributes and comments of attitude and request
        overview_output, attitude_list, request_list = generate_input_text(overview_df, attitude_df, request_df)
        logger.info("Generating input text from data")

        # # 3) load model for prediction
        # model, tokenizer = predict_LLAMA2.load_LLAMA2_model(model_dir="models/llama2/")
        # logger.info("Model loaded successfully.")

        # 4) just reuse the finalized overview data
        summary = []
        summary.append("#### Overview")
        summary.append(overview_output)
        #for line in overview_output.splitlines():
        #    summary.append(line)

        # 5) add given Attitude Roots and their AI generated summary of comments
        logger.info("Processing Attitude Roots...")
        summary.append("")
        summary.append ("#### Attitude Roots")
        attitude_roots, comments = attitude_list

        # case of no return what means that there arent any attitude roots
        if attitude_roots == [] and comments == []:
            summary.append("No attitude roots were found during analysis.")

        # case of return what means we can predict!
        for attitude, comment in zip(attitude_roots, comments):
            pred = predict_LLAMA2.predict(comment, model=model, tokenizer=tokenizer)
            summary.append(f"{attitude}  \n *AI aggregated comments*: {pred}")


        # 6) add given Reqest Information and their AI generated summary of comments
        logger.info("Processing Request Information...")
        summary.append("")
        summary.append("#### Request Information")
        requests, comments = request_list

        # case of no return what means that there arent any attitude roots
        if requests == [] and comments == []:
            summary.append("No request information were found during analysis.")

        # case of return what means we can predict!
        for request, comment in zip(requests, comments):
            pred = predict_LLAMA2.predict(comment, model=model, tokenizer=tokenizer)
            summary.append(f"{request}  \n *AI aggregated comments*: {pred}")

        summary_df = pd.DataFrame(summary)
        
        # # TODO: remove this line when we use the model for prediction
        # # summary_df = pd.DataFrame(["## Temporary dummy", "We currently commented out AI summary generation. Go to DASP_report_template/summary_generator/main.py to use it again."])
        
        summary_df_dict = summary_df.to_dict(orient="records")
        logger.info("Summary generation completed successfully.")
        
        # with open("summary.json", "r") as f:
        #     summary_df_dict = json.load(f)

        
        return summary_df_dict
    
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
    
# Run the application on port 8083
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8083)