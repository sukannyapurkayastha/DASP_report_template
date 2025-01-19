"""
prediction_BART.py

A script that:
1) Loads a previously fine-tuned BART model from ./models/bart
2) Predicts summary of input
"""

import sys
import logging
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_BART_model(model_dir: str = "./models/bart"):
    """
    Load BART model previously trained by train_BART.py
      
      Args:
          model_dir (str): path to directory where the model is supposed to be stored.
    """

    logger.info(f"Loading BART model from: {model_dir}")
    tokenizer = BartTokenizer.from_pretrained(model_dir)
    model = BartForConditionalGeneration.from_pretrained(model_dir)
    
    return model, tokenizer

def predict(
    input_text: str,
    model = None,
    tokenizer = None,
    model_dir: str = "./models/bart",
    min_new_tokens: int = 20,
) -> str:
    """
    Generates a BART prediction such that the output is around
    two-thirds the token length of the input text.

    Args:
        input_text (str): The raw input text to be summarized/transformed.
        model_dir (str): Path to the directory where the BART model is stored.
        min_new_tokens (int): A minimum number of tokens to generate
                              (avoids extremely short outputs).

    Returns:
        A string containing the BART model's output.
    """

    # 1) Load tokenizer and model
    if (model == None and tokenizer == None):
        model, tokenizer = load_BART_model(model_dir)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2) Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    #input_length = inputs["input_ids"].shape[1]

    # 3) Calculate desired output length (~2/3 of input length)
    #desired_output_length = int(round((2.0 / 3.0) * input_length))
    # Ensure it's at least `min_new_tokens` to avoid producing almost nothing
    #desired_output_length = max(desired_output_length, min_new_tokens)

    #logger.info(f"Detected input length (tokens): {input_length}")
    #logger.info(f"Desired output length (tokens): {desired_output_length}")

    # 4) Generate
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # We use `max_new_tokens=desired_output_length` to limit generation
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=150,
        num_beams=4,               # optional: beam search
        no_repeat_ngram_size=2     # optional: avoid repeating phrases
    )

    # 5) Decode
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction

if __name__ == "__main__":
    
    # 1) get dummy data
    import dfs_to_input_converter
    dummy_input_data = dfs_to_input_converter.generate_dummy_input_text()
    
    # 2) generate BART output
    pred = predict(input_text=dummy_input_data)
    print("BART Output:")
    print(pred)
