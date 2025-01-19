import os
import pandas as pd
import ast
#from dfs_to_input_converter import generate_input_text

def process_csv_files(base_path="data/real_sample_data"):
    # Dictionary to hold dataframes for each folder, with each folder containing a list of DataFrames
    folder_dataframes = {}

    # Iterate through folders
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        
        # Ensure it's a directory
        if os.path.isdir(folder_path):
            # Ensure specific order of CSV files
            csv_order = ["overview.csv", "attitude_roots.csv", "request_information.csv"]
            
            folder_csvs = []  # List to store DataFrames for this folder
            
            for csv_file in csv_order:
                file_path = os.path.join(folder_path, csv_file)

                if os.path.exists(file_path):
                    # Read CSV file into DataFrame
                    df = pd.read_csv(file_path)

                    # Parse the 'Individual_scores' column (if present)
                    if 'Individual_scores' in df.columns:
                        df['Individual_scores'] = df['Individual_scores'].apply(lambda x: ast.literal_eval(x))

                    # Parse and adjust 'Comments' column (if present)
                    if 'Comments' in df.columns:
                        df['Comments'] = df['Comments'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

                    # Round numerical columns (if any)
                    for col in df.select_dtypes(include=['float', 'int']).columns:
                        df[col] = df[col].round(2)

                    folder_csvs.append(df)

            folder_dataframes[folder] = folder_csvs

    return folder_dataframes


def convert_dataframes_to_input(folder_dataframes):
    # Dictionary to store input strings for each folder
    folder_inputs = {}

    for folder, dfs in folder_dataframes.items():
        if len(dfs) == 3:
            overview_df, attitude_df, request_df = dfs
            # Generate input text using the imported function
            input_text = generate_input_text(overview_df, attitude_df, request_df)
            folder_inputs[folder] = input_text

    return folder_inputs

###############################################################################
#                      MODEL-INPUT-BUILDING FUNCTIONS
###############################################################################

def generate_overview_prompts(overview_df):
    """
    Build lines summarizing rating, soundness, etc.
    """
    max_scores = {'Rating': 10, 'Soundness': 4, 'Presentation': 4, 'Contribution': 4}
    threshold = 1.5
    overview_sentences = []

    for _, row in overview_df.iterrows():
        category = row['Category']
        avg_score = row['Avg_Score']
        individual_scores = row['Individual_scores']
        max_score = max_scores.get(category, "Unknown")

        scores = [score for (_, score) in individual_scores if score is not None]
        outliers = [sc for sc in scores if abs(sc - avg_score) > threshold]

        sentence = f"- {category} is {avg_score} out of {max_score}."
        if outliers:
            unique_outliers = sorted(set(outliers))
            if len(unique_outliers) == 1:
                outliers_text = f"a rating at {unique_outliers[0]}"
            else:
                outliers_text = ", ".join([f"a rating at {sc}" for sc in unique_outliers[:-1]])
                outliers_text += f" and a rating at {unique_outliers[-1]}"
            sentence += f" Outlier was {outliers_text}."

        overview_sentences.append(sentence)

    return "\n".join(overview_sentences)

def generate_attitude_roots_prompts(attitude_df):
    """
    Summarize the attitude roots, frequencies, and comments.
    """
    attitude_df['Frequency_Percent'] = attitude_df['Frequency'] * 100
    lines = []
    comments = []

    for _, row in attitude_df.iterrows():
        root = row['Attitude_roots']
        freq = row['Frequency_Percent']
        descr = row['Descriptions']
        comments_list = row['Comments']

        combined_comments = []
        for _, comment_chunk in comments_list:
            combined_comments.append(" ".join(comment_chunk))

        all_comments = " ".join(combined_comments)
        line = f"- {root} appears {freq:.0f}% of the time. {descr}."
        lines.append(line)
        comments.append(all_comments)

    

    return [lines, comments]

def generate_request_information_prompts(request_df):
    """
    Summarize requests (like "Typo" or "Clarification"), freq, and comments.
    """
    request_df['Frequency_Percent'] = request_df['Frequency'] * 100
    lines = []
    comments = []

    for _, row in request_df.iterrows():
        request_type = row['Request Information']
        freq = row['Frequency_Percent']
        comments_list = row['Comments']

        combined_comments = []
        for _, chunk in comments_list:
            combined_comments.append(" ".join(chunk))

        all_comments = " ".join(combined_comments)
        line = f"- {request_type} was requested {freq:.0f}% of the time."
        lines.append(line)
        comments.append(all_comments)

    return [lines, comments]


def generate_input_text(overview_df, attitude_df, request_df):
    
    # Collect all prompts from the three scripts
    overview_output = generate_overview_prompts(overview_df)
    attitude_list = generate_attitude_roots_prompts(attitude_df)
    request_list = generate_request_information_prompts(request_df)

    # Combine all data into a single prompt with clear section headings

    return overview_output, attitude_list, request_list


###############################################################################
#                      DATA EXPORT FUNCTIONS
###############################################################################

def export_unlabeled_data():
    complete_export = ""
    folder_dataframes = process_csv_files() # return a list [overview_df, attitude_df. request_df]
    # dfs = [overview_df, attitude_df, request_df]
    for folder, dfs in folder_dataframes.items():
        overview_df, attitude_df, request_df = dfs
        overview_output, attitude_list, request_list = generate_input_text(overview_df, attitude_df, request_df)
        
        # export each comment
        attitude_roots, comments = attitude_list
        for comment in comments:
            part_string_for_export = '{"input": "' + f"{comment}" + '"}\n'
            complete_export += part_string_for_export
         
        request, comments = request_list
        for comments in comments:
            part_string_for_export = '{"input": "' + f"{comments}" + '"}\n'
            complete_export += part_string_for_export
            
    print(complete_export)        
    with open("real_world_data_unlabeled.jsonl", "w", encoding="utf-8") as output_file:
        output_file.write(complete_export)

    print("Predictions have been exported to real_word_data_unlabeled.jsonl")
    
if __name__ == "__main__":
    export_unlabeled_data()
    

def predict_data(model, tokenizer, overview_df, attitude_df, request_df):
    
    from data.convert_csv_to_unlabeled_jsonl import generate_input_text
    
    overview_output, attitude_list, request_list = generate_input_text(overview_df, attitude_df, request_df)
    
    # 1) Write Overview part without AI
    summary = "Overview: \n"
    summary += overview_output
    
    # 2) add AI generated summary of Attitue Roots
    summary += "\n Attitude Roots: \n"
    attitude_roots, comments = attitude_list
    for attitude, comment in attitude_roots, comments:
        pred = model.predict(comment, tokenizer) #TODO:
        summary += f"{attitude} AI aggregated Comments: {pred} \n"
    
    # 3) add AI generated summary of Request Information
    summary += "\n Request Information: \n"
    requests, comments = request_list
    for request, comment in requests, comments:
        pred = model.predict(comment) #TODO:
        summary += f"{request} AI aggregated Comments: {pred} \n"
        