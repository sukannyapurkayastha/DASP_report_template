from loguru import logger

import pandas as pd
import torch
from torch.nn import Sigmoid
import os
from description_generation import extract_sentences, get_most_representative_sentence
from transformers import BertTokenizer, BertForSequenceClassification

# from utils import load_DistilBertTokenizer, load_TFDistilBertForSequenceClassification, load_BertTokenizer, \
#     load_BertForSequenceClassification


def predict_root_category(text, model, tokenizer):
    """
    Predicts the root category of a given text using a pre-trained BERT model.

    Args:
        text (str): The input text for classification.
        model (BertForSequenceClassification): Pre-trained BERT model for root category classification.
        tokenizer (BertTokenizer): Tokenizer corresponding to the BERT model.

    Returns:
        int: The predicted category index.
    """

    predict_input = tokenizer.encode(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    try:
        with torch.no_grad():
            outputs = model(predict_input)
            logits = outputs.logits
        prediction_value = torch.argmax(logits, dim=1).item()
    except Exception as e:
        logger.error(e)

    return prediction_value


def attitude_roots_prediction(data):
    """
    Predicts attitude root categories for given textual data.

    Args:
        data (pd.DataFrame): A DataFrame containing a 'sentence' column.

    Returns:
        pd.DataFrame: The input DataFrame with additional columns:
                      - 'attitude_root_number': The predicted root category index.
                      - 'attitude_root': The corresponding category label.
    """
    local_path = "models/attitude_root/"
    huggingface_model_path = "DASP-ROG/AttitudeModel"

    # Load the tokenizer and model from huggingface if not available locally
    logger.info("Loading tokenizer and model for root category ...")
    tokenizer = BertTokenizer.from_pretrained(huggingface_model_path, cache_dir=local_path)
    model = BertForSequenceClassification.from_pretrained(huggingface_model_path, num_labels=9,
                                                          cache_dir=local_path)
    model.eval()

    logger.info("Predicting root category")
    data['attitude_root_number'] = data['sentence'].apply(lambda x: predict_root_category(x, model, tokenizer))
    logger.info(f"Root category prediction done.")

    label_mapping = {
        0: 'None',
        1: 'Substance',
        2: 'Originality',
        3: 'Clarity',
        4: 'Soundness-correctness',
        5: 'Motivation-impact',
        6: 'Meaningful-comparison',
        7: 'Replicabilitye',
        8: 'Other'
    }
    
    data['attitude_root'] = data['attitude_root_number'].map(label_mapping)
    data = data[data['attitude_root'] != 'None']

    return data


def predict_theme_category(text, model, tokenizer):
    """
    Predicts the theme category of a given text using a pre-trained BERT model.

    Args:
        text (str): The input text for classification.
        model (BertForSequenceClassification): Pre-trained BERT model for theme classification.
        tokenizer (BertTokenizer): Tokenizer corresponding to the BERT model.

    Returns:
        list[str]: A list of predicted theme labels.
    """
    threshold = 0.5

    # Tokenize the input text using tokenizer (handles padding, truncation, etc.)
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

    # Run the model for prediction
    with torch.no_grad():  # Disable gradient calculation during inference
        outputs = model(**inputs)
        logits = outputs.logits

    # Apply sigmoid to logits (since it's multi-label classification)
    sigmoid = Sigmoid()
    probabilities = sigmoid(logits)

    # Apply threshold to get binary predictions (0 or 1)
    predictions = (probabilities > threshold).int()
    predictions = predictions.squeeze().tolist()

    labels = ['ANA', 'BIB', 'DAT', 'EXP', 'INT', 'MET', 'OAL', 'PDI', 'RES', 'RWK', 'TNF']
    # Find all indices of 1
    indices = [i for i, value in enumerate(predictions) if value == 1]
    true_labels = [labels[i] for i in indices]

    # Return the binary predictions (as a list)
    return true_labels


def create_clusters(row):
    """
    Combines root category with theme labels to create clusters.

    Args:
        row (pd.Series): A row from a DataFrame containing 'attitude_root' and 'attitude_themes'.

    Returns:
        list[str]: A list of combined root and theme clusters.
    """
    # Extract root and themes
    root = row["attitude_root"]
    themes = row["attitude_themes"]

    # Combine root with each theme
    clusters = [f"{root}({theme})" for theme in themes]
    return clusters


def combine_roots_and_themes(preprocessed_data):
    """
    Combines root category predictions with theme category predictions.

    Args:
        preprocessed_data (pd.DataFrame): A DataFrame containing sentences to be classified.

    Returns:
        pd.DataFrame: A DataFrame containing:
                      - 'Attitude_roots': Combined root and theme clusters.
                      - 'Frequency': Frequency of each cluster occurrence.
                      - 'Descriptions': Descriptions of each category.
                      - 'Comments': Aggregated comments associated with each category.
    """
    df = attitude_roots_prediction(preprocessed_data)

    # Load the pretrained model and tokenizer
    local_path = "models/attitude_theme/"
    huggingface_model_path = "DASP-ROG/ThemeModel"
    tokenizer = BertTokenizer.from_pretrained(huggingface_model_path, cache_dir=local_path)
    model = BertForSequenceClassification.from_pretrained(huggingface_model_path, num_labels=11, cache_dir=local_path,
                                                          problem_type="multi_label_classification")

    model.eval()
    # theme prediction
    logger.info("Predicting attitude theme category")
    df['attitude_themes'] = df['sentence'].apply(lambda x: predict_theme_category(x, model, tokenizer))  # attitude_themes_prediction
    logger.info("Attitude theme prediction done")

    # Apply the function to create clusters
    df.loc[:, "clusters"] = df.apply(create_clusters, axis=1)
    df = df.explode("clusters", ignore_index=True)

    # Count distinct authors
    distinct_authors_count = df['author'].nunique()
    # Group by 'author' and 'clusters', and aggregate the sentences into a list
    aggregated_df = df.groupby(['clusters', 'author'])['sentence'].apply(list).reset_index()
    final_df = aggregated_df.groupby('clusters').agg(
        comments=('sentence', lambda x: [[author, sentences] for author, sentences in zip(aggregated_df['author'], x)])
    ).reset_index()
    final_df['Frequency'] = final_df['comments'].apply(len) / distinct_authors_count
    final_df['Descriptions'] = 'none'
    final_df = final_df.rename(columns={'comments': 'Comments', 'clusters': 'Attitude_roots'})
    final_df = final_df[['Attitude_roots', 'Frequency', 'Descriptions', 'Comments']]
    final_df = final_df.sort_values(by='Frequency', ascending=False)

    current_path = os.path.dirname(os.path.abspath(__file__))
    desc = pd.read_csv(os.path.join(current_path, 'attitudes_desc.csv'))

    merged_df = pd.merge(final_df, desc, on=['Attitude_roots'],
                         how='left')  # todo:what happens if attitude + theme combi is not known
    merged_df.rename(columns={'Descriptions_y': 'Descriptions'}, inplace=True)

    # Drop Descriptions_x column
    merged_df.drop(columns=['Descriptions_x'], inplace=True)

    # generate description if there is no corresponding description
    if merged_df['Descriptions'].isna().any():
        logger.info('New attitude clusters appear, generating description...')
        # Iterate through the rows and generate description for missing descriptions
        for index, row in merged_df.iterrows():
            if pd.isna(row['Descriptions']):
                attitude_root = row['Attitude_roots']
                logger.info(f'New attitude clusters appear: {attitude_root}')
                # Generate descriptions if it's missing
                comments = row['Comments']  
                input_texts = extract_sentences(comments)
                logger.info(f'input sentences: {input_texts}')
                best_sentence = get_most_representative_sentence(input_texts)
                logger.info(f'desc: {best_sentence}')
                
                # Set the best sentence as the description
                merged_df.at[index, 'Descriptions'] = best_sentence

                # Update the corresponding entry in desc DataFrame with the new attitude_root and description
                new_row = pd.DataFrame({'Attitude_roots': [attitude_root], 'Descriptions': [best_sentence]})
                desc = pd.concat([desc, new_row], ignore_index=True)
        # Save the updated desc DataFrame back to CSV
        logger.info('New attitude clusters appear, update attitudes_desc.csv')
        desc.to_csv(os.path.join(current_path, 'attitudes_desc.csv'), index=False)

    # change order as expected from frontend
    merged_df = merged_df[['Attitude_roots', 'Frequency', 'Descriptions', 'Comments']]
    return merged_df

    # label_mapping = {
    #     0: 'Other',
    #     1: 'Clarity',
    #     2: 'Meaningful-comparison',
    #     3: 'Motivation-impact',
    #     4: 'Originality',
    #     5: 'Replicability',
    #     6: 'Soundness-correctness',
    #     7: 'Substance',
    #     8: 'None'
    # }

    # # match dataframe schema
    # final_df['Attitude_roots'] = final_df['attitude_root_number'].map(label_mapping)
    # final_df['Descriptions'] = 'none'
    # final_df.drop(columns=['attitude_root_number'], inplace=True)
    # final_df = final_df.rename(columns={'comments': 'Comments'})
    # final_df['Frequency'] = final_df['count'] / total_rows
    # final_df.drop(columns=['count'], inplace=True)
    # final_df = final_df[['Attitude_roots', 'Frequency', 'Descriptions', 'Comments']]
