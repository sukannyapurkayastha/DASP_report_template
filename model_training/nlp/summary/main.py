"""
 main.py

This file 
(1) checks wheather cuda is availibile 
(2) executes the model training for all models we want to investigate and require training, 
(3.1) computes evaluation measures for each model using a test-set and saves it into a model_comparison.csv
(3.2) makes a prediciton for a dummy data set for each model to make it more human readable and saves the results into output.txt  

"""
import verify_cuda
import train_T5, train_BART, train_BLOOM
import predict_compare
import predict_T5, predict_BART, predict_BLOOM, predict_LLAMA2
import dfs_to_input_converter, input_to_prompt_converter
from data.data_processing import generate_input_text
import dummy_data




# 1) Verify CUDA is available
verify_cuda.main()

# 2) Train models (if not already done)
train_T5.main()
train_BART.main()
train_BLOOM.main()
# llama2 is not trained as it is already very good.


# 3) Generate test predictions for each model

# 3.1) Compare model predictions by metric using a test-set
predict_compare.main()

# 3.2) Make a specific prediction and wirte it to a file for human comparison
# 3.2.1) Dummy data
dummy_overview_df, dummy_attitude_df, dummy_request_df = dummy_data.get_dummy_data()
overview_output, attitude_list, request_list = generate_input_text(dummy_overview_df, dummy_attitude_df, dummy_request_df)


# 3.2.2) load all models predictions
model_t5, tokenizer_t5 = predict_T5.load_T5_model()
model_bart, tokenizer_bart = predict_BART.load_BART_model()
model_bloom, tokenizer_bloom = predict_BLOOM.load_BLOOM_model()
model_llama2, tokenizer_llama2 = predict_LLAMA2.load_LLAMA2_model()


# make predictions for each model

# %%T5
# 1) Write Overview part without AI
summary_t5 = "Overview: \n"
summary_t5 += overview_output

# 2) add AI generated summary_t5 of Attitue Roots
summary_t5 += "\n Attitude Roots: \n"
attitude_roots, comments = attitude_list
for attitude, comment in zip(attitude_roots, comments):
    pred = predict_T5.predict(comment, model=model_t5, tokenizer=tokenizer_t5)
    summary_t5 += f"{attitude} \nAI aggregated comments: {pred} \n\n"

# 3) add AI generated summary_t5 of Request Information
summary_t5 += "\n Request Information: \n"
requests, comments = request_list
for request, comment in zip(requests, comments):
    pred = predict_T5.predict(comment, model=model_t5, tokenizer=tokenizer_t5)
    summary_t5 += f"{request} \nAI aggregated comments: {pred} \n\n"

# %% BART
summary_bart = "Overview: \n"
summary_bart += overview_output

# 2) add AI generated summary_bart of Attitue Roots
summary_bart += "\n Attitude Roots: \n"
attitude_roots, comments = attitude_list
for attitude, comment in zip(attitude_roots, comments):
    pred = predict_BART.predict(comment, model=model_bart, tokenizer=tokenizer_bart)
    summary_bart += f"{attitude} \nAI aggregated comments: {pred} \n\n"

# 3) add AI generated summary_bart of Request Information
summary_bart += "\n Request Information: \n"
requests, comments = request_list
for request, comment in zip(requests, comments):
    pred = predict_BART.predict(comment, model=model_bart, tokenizer=tokenizer_bart)
    summary_bart += f"{request} \nAI aggregated comments: {pred} \n\n"

#%% BLOOM
summary_bloom = "Overview: \n"
summary_bloom += overview_output

# 2) add AI generated summary_bloom of Attitue Roots
summary_bloom += "\n Attitude Roots: \n"
attitude_roots, comments = attitude_list
for attitude, comment in zip(attitude_roots, comments):
    pred = predict_BLOOM.predict(comment, model=model_bloom, tokenizer=tokenizer_bloom)
    summary_bloom += f"{attitude} \nAI aggregated comments: {pred} \n\n"

# 3) add AI generated summary_bloom of Request Information
summary_bloom += "\n Request Information: \n"
requests, comments = request_list
for request, comment in zip(requests, comments):
    pred = predict_BLOOM.predict(comment, model=model_bloom, tokenizer=tokenizer_bloom)
    summary_bloom += f"{request} \nAI aggregated comments: {pred} \n\n"

#%% llama2 
summary_llama2 = "Overview: \n"
summary_llama2 += overview_output

# 2) add AI generated summary_llama2 of Attitue Roots
summary_llama2 += "\n Attitude Roots: \n"
attitude_roots, comments = attitude_list
for attitude, comment in zip(attitude_roots, comments):
    pred = predict_LLAMA2.predict(comment, model=model_llama2, tokenizer=tokenizer_llama2)
    summary_llama2 += f"{attitude} \nAI aggregated comments: {pred} \n\n"

# 3) add AI generated summary_llama2 of Request Information
summary_llama2 += "\n Request Information: \n"
requests, comments = request_list
for request, comment in zip(requests, comments):
    pred = predict_LLAMA2.predict(comment, model=model_llama2, tokenizer=tokenizer_llama2)
    summary_llama2 += f"{request} \nAI aggregated comments: {pred} \n\n"


predictions = f"""
-------------------------------------------------------------------
Final model Outputs:

T5-Output:
{summary_t5}

BART-Output:
{summary_bart}

BLOOM-Output:
{summary_bloom}


LLAMA2-Output:
{summary_llama2}
-------------------------------------------------------------------
"""

# 3.2.3) Export as output.txt to data/output.txt
with open("data/output.txt", "w", encoding="utf-8") as output_file:
    output_file.write(predictions)

print("Predictions have been exported to data/output.txt")
