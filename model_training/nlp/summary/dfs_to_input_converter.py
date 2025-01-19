# dfs_to_input_converter.py


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

    for _, row in attitude_df.iterrows():
        root = row['Attitude_roots']
        freq = row['Frequency_Percent']
        descr = row['Descriptions']
        comments_list = row['Comments']

        combined_comments = []
        for _, comment_chunk in comments_list:
            combined_comments.append(" ".join(comment_chunk))

        all_comments = " ".join(combined_comments)
        line = f"- {root} appears {freq:.0f}% of the time. {descr}. Comments: {all_comments}"
        lines.append(line)

    return "\n".join(lines)

def generate_request_information_prompts(request_df):
    """
    Summarize requests (like "Typo" or "Clarification"), freq, and comments.
    """
    request_df['Frequency_Percent'] = request_df['Frequency'] * 100
    lines = []

    for _, row in request_df.iterrows():
        request_type = row['Request Information']
        freq = row['Frequency_Percent']
        comments_list = row['Comments']

        combined_comments = []
        for _, chunk in comments_list:
            combined_comments.append(" ".join(chunk))

        all_comments = " ".join(combined_comments)
        line = f"- {request_type} was requested {freq:.0f}% of the time. Comments: {all_comments}"
        lines.append(line)

    return "\n".join(lines)


def generate_input_text(overview_df, attitude_df, request_df):
    
    # Collect all prompts from the three scripts
    overview_prompts = generate_overview_prompts(overview_df)
    attitude_prompts = generate_attitude_roots_prompts(attitude_df)
    request_prompts = generate_request_information_prompts(request_df)

    # Combine all data into a single prompt with clear section headings
    input_data = f"""
Overview:
{overview_prompts}

Attitude Roots:
{attitude_prompts}

Request Information:
{request_prompts}
    """
    return input_data


def generate_dummy_input_text():
    """
    Returns dummy input data string for debug purposes
    """
    import dummy_data
    
    # Import dummy data from respective module
    overview_df, attitude_df, request_df = dummy_data.get_dummy_data()
    
    # Translate dummy data into input string expected
    dummy_input_data = generate_input_text(overview_df, attitude_df, request_df)
    
    return dummy_input_data
