from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from loguru import logger


# Load the pretrained model and tokenizer
local_path = "/opt/models/attitude/description_generator"
huggingface_model_path = "DASP-ROG/DescriptionModel"
tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path, cache_dir=local_path)
model = AutoModelForSeq2SeqLM.from_pretrained(huggingface_model_path, cache_dir=local_path, torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def get_most_representative_sentence(sentences):
    """
    Find the sentence with the highest average cosine similarity to all other sentences.
    
    :param sentences: List of input sentences.
    :return: The most representative sentence.
    """
    # Tokenize and generate outputs for all sentences without decoding
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=512)
    input_ids = inputs.input_ids.to(device)

    # Generate outputs using the T5 model (tokenized output)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=200, num_beams=4, length_penalty=2.0)

    # Now, we work directly with the tokenized output_ids
    embeddings = np.vstack([output_ids.cpu().numpy() for output in output_ids])  # Get the tokenized outputs

    # Compute cosine similarity matrix for the tokenized outputs
    similarity_matrix = cosine_similarity(embeddings)  # Shape: (num_sentences, num_sentences)
    logger.info(similarity_matrix)
    
    # Compute the average similarity for each sentence
    avg_similarities = similarity_matrix.mean(axis=1)
    
    # Find the index of the most representative sentence
    best_sentence_idx = np.argmax(avg_similarities)
    
    # Decode the best sentence from the output_ids
    best_sentence = tokenizer.decode(output_ids[best_sentence_idx], skip_special_tokens=True)
    
    return best_sentence

# def get_embedding(text):
#     """Compute the SciBERT embedding by averaging over the last layer representations."""
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
#     with torch.no_grad():  # No gradient calculation needed for inference
#         outputs = model(**inputs)
    
#     last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_dim)
#     sentence_embedding = last_hidden_state.mean(dim=1)  # Averaging over all tokens
    
#     return sentence_embedding.cpu().numpy()  # Convert to NumPy array

# def get_most_representative_sentence(sentences):
#     """Find the sentence with the highest average cosine similarity to all other sentences."""
#     embeddings = np.vstack([get_embedding(sent) for sent in sentences])  # Get embeddings for all sentences
    
#     # Compute cosine similarity matrix
#     similarity_matrix = cosine_similarity(embeddings)  # Shape: (num_sentences, num_sentences)
    
#     # Compute the average similarity for each sentence
#     avg_similarities = similarity_matrix.mean(axis=1)
    
#     # Find the index of the most representative sentence
#     best_sentence_idx = np.argmax(avg_similarities)
    
#     return sentences[best_sentence_idx]


# def generate_descriptions(input_texts):
#     """
#     Generates descriptions using the pretrained model for a list of sentences.
    
#     :param input_texts: List of input sentences.
#     :return: List of generated descriptions.
#     """
    
#     # Tokenize inputs (batch size = len(input_texts))
#     inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
#     input_ids = inputs.input_ids.to(device)

#     # Generate outputs
#     with torch.no_grad():
#         output_ids = model.generate(input_ids, max_length=200, num_beams=4, length_penalty=2.0)

#     # Decode output
#     output_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in output_ids]
    
#     return output_texts

def extract_sentences(data):
    """
    Extracts all sentences from a nested list while ignoring reviewer names.

    :param data: List of lists where each sublist contains a reviewer name and a list of sentences.
    :return: A flattened list of all sentences.
    """
    return [sentence for item in data for sentence in item[1]]

def generate_desc(df):
    """
    Generate descriptions for the 'Descriptions' column in the DataFrame.
    
    :param df: DataFrame containing 'Attitude_roots' and 'Descriptions'.
    :return: DataFrame with generated descriptions.
    """
    # Iterate through the rows and generate description for missing descriptions
    for index, row in df.iterrows():
        if pd.isna(row['Descriptions']) or row['Descriptions'] == '':
            # Generate descriptions if it's missing or empty
            comments = row['Comments']  
            input_texts = extract_sentences(comments)
            logger.info(f'input sentences: {input_texts}')
            best_sentence = get_most_representative_sentence(input_texts)
            logger.info(f'desc: {best_sentence}')
            
            # Set the best sentence as the description
            df.at[index, 'Descriptions'] = best_sentence
    
    return df

# if __name__ == "__main__":
#     data = pd.read_pickle('test_data.pkl')
#     result = generate_desc(data)
#     result.to_pickle('test_data_result.pkl')