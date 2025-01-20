from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn
from data_processing import generate_input_text, predict_data
import predict_LLAMA2

app = FastAPI()

# Define schema for incoming preprocessed data
class RawInput(BaseModel):
    data: list[dict]

@app.post("/generate_summary")
async def predict(overview_df: RawInput, attitude_df: RawInput, request_df: RawInput) -> list[dict]:
    try:
        # 1) Transform json files into dfs
        overview = overview_df.data
        overview_df = pd.DataFrame(overview)
        
        attitude = attitude_df.data
        attitude_df = pd.DataFrame(attitude)
        
        request = request_df.data
        request_df = pd.DataFrame(request)

        
        # 2) Transform overview into its final text and extract lists with attributes and comments of attitude and request 
        overview_output, attitude_list, request_list = generate_input_text(overview_df, attitude_df, request_df)
        
        # 3) load model for prediction 
        model, tokenizer = predict_LLAMA2.load_LLAMA2_model()
        
        # 4) just use the finalized overview data
        summary = "Overview: \n"
        summary += overview_output


        # 5) add given Attitude Roots and their AI generated summary of comments
        summary += "\n Attitude Roots: \n"
        attitude_roots, comments = attitude_list
        
        # case of no return what means that there arent any attitude roots
        if attitude_roots == [] and comments == []:
            summary += "No attitude roots were found during analysis."
        
        # case of return what means we can predict!
        for attitude, comment in zip(attitude_roots, comments):
            pred = predict_LLAMA2.predict(comment, model=model, tokenizer=tokenizer)
            summary += f"{attitude} \nAI aggregated comments: {pred} \n\n"


        # 6) add given Reqest Information and their AI generated summary of comments
        summary += "\n Request Information: \n"
        requests, comments = request_list
        
        # case of no return what means that there arent any attitude roots
        if requests == [] and comments == []:
            summary += "No request information were found during analysis."
            
        # case of return what means we can predict!
        for request, comment in zip(requests, comments):
            pred = predict_LLAMA2.predict(comment, model=model, tokenizer=tokenizer)
            summary += f"{request} \nAI aggregated comments: {pred} \n\n"
        
        # turn string into a pandas df
        summary_df = pd.DataFrame([[summary]])
        
        summary_df_dict = summary_df.to_dict(orient="records")
        
        return summary_df_dict
    
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
    
# Run the application on port 8083
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8083)