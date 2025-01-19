"""
prediction_BLOOM.py

A script that:
1) Loads a previously fine-tuned BLOOM model from ./models/bloom
2) Prints ONLY the newly generated text (omitting the prompt)
"""

import os
import logging
import torch
from transformers import BloomTokenizerFast, BloomForCausalLM
import input_to_prompt_converter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_BLOOM_model(model_dir: str = "./models/bloom"):
    """
    Load BLOOM model previously trained by train_BLOOM.py
      
      Args:
          model_dir (str): path to directory where the model is supposed to be stored.
    """

    logger.info(f"Loading BLOOM model from: {model_dir}")
    tokenizer = BloomTokenizerFast.from_pretrained(model_dir)
    model = BloomForCausalLM.from_pretrained(model_dir)
    
    # Option 1: Set pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Option 2: Add a new pad_token (if preferred)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model.resize_token_embeddings(len(tokenizer))
    # model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def predict(
    input_text: str,
    model=None,
    tokenizer=None,
    model_dir: str = "./models/bloom",
    min_new_tokens: int = 20,
) -> str:
    """
    Generates a BLOOM prediction such that the output is around
    two-thirds the token length of the (prompted) input text,
    and returns ONLY the newly generated tokens (excluding the prompt).
    """

    # 1) Load tokenizer and model if not provided
    if (model is None and tokenizer is None):
        model, tokenizer = load_BLOOM_model(model_dir)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2) Build and tokenize the input
    prompt_tokens = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
    prompt_length = prompt_tokens["input_ids"].shape[1]


    # 4) Generate text
    input_ids = prompt_tokens["input_ids"].to(device)
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=150,
        num_beams=4,
        no_repeat_ngram_size=2
    )

    # 5) Decode only the newly generated tokens
    full_sequence = outputs[0]                     # shape: [total_length]
    new_tokens = full_sequence[prompt_length:]     # slice off the prompt
    prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return prediction.strip()

if __name__ == "__main__":
    import dfs_to_input_converter
    dummy_input_data = dfs_to_input_converter.generate_dummy_input_text()

    pred = predict(input_text=dummy_input_data)
    print("BLOOM (newly generated) Output:")
    print(pred)
