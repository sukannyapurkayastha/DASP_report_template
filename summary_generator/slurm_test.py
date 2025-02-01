import json
import pandas as pd
from data_processing import generate_input_text, predict_data
import predict_LLAMA2
from loguru import logger


def final_predict() -> list[dict]:
    try:
        # 1) Transform json files into dfs
        # /home/nana/DASP_report_template/attitude_roots.csv
        overview_df = pd.read_pickle('/home/nana/DASP_report_template/overview.pkl')
        
        attitude_df = pd.read_pickle('/home/nana/DASP_report_template/attitude_roots.pkl')
        
        request_df = pd.read_pickle('/home/nana/DASP_report_template/request_information.pkl')

        # 2) Transform overview into its final text and extract lists with attributes and comments of attitude and request
        overview_output, attitude_list, request_list = generate_input_text(overview_df, attitude_df, request_df)

        # 3) load model for prediction
        model, tokenizer = predict_LLAMA2.load_LLAMA2_model(model_dir="models/llama2/")

        # 4) just reuse the finalized overview data
        summary = []
        summary.append("#### Overview")
        for attribute in overview_df.iterrows():
            summary.append(overview_output)

        # 5) add given Attitude Roots and their AI generated summary of comments
        summary.append("")
        summary.append ("#### Attitude Roots")
        attitude_roots, comments = attitude_list

        # case of no return what means that there arent any attitude roots
        if attitude_roots == [] and comments == []:
            summary.append("No attitude roots were found during analysis.")

        # case of return what means we can predict!
        for attitude, comment in zip(attitude_roots, comments):
            pred = predict_LLAMA2.predict(comment, model=model, tokenizer=tokenizer)
            summary.append(f"- {attitude}  \n **AI aggregated comments**: {pred}")


        # 6) add given Reqest Information and their AI generated summary of comments
        summary.append("")
        summary.append("#### Request Information")
        requests, comments = request_list

        # case of no return what means that there arent any attitude roots
        if requests == [] and comments == []:
            summary.append("No request information were found during analysis.")

        # case of return what means we can predict!
        for request, comment in zip(requests, comments):
            pred = predict_LLAMA2.predict(comment, model=model, tokenizer=tokenizer)
            summary.append(f"- {request}  \n **AI aggregated comments**: {pred}")

        summary_df = pd.DataFrame(summary)
        
        # TODO: remove this line when we use the model for prediction
        # summary_df = pd.DataFrame(["## Temporary dummy", "We currently commented out AI summary generation. Go to DASP_report_template/summary_generator/main.py to use it again."])
        
        summary_df_dict = summary_df.to_dict(orient="records")
        
        return summary_df_dict
    
    
    except Exception as e:
        logger.exception("An error occurred!")

if __name__ == "__main__":
    summary_df_dict = final_predict()
    with open("summary.json", "w") as f:
        json.dump(summary_df_dict, f, indent=4)