"""
predict_LLAMA2.py

A script that:
1) Loads a LLaMA2 model from Hugging Face
2) Predicts summary of input
"""

import pandas as pd
import torch
import logging
import sys
import os

import input_to_prompt_converter
from transformers import (
    GenerationConfig,
)

# Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


def predict(
        input_text: str,
        model=None,
        tokenizer=None,
        model_dir: str = "/storage/ukp/shared/shared_model_weights/models--llama-2-hf/7B-Chat",
        min_new_tokens: int = 20,
        max_new_tokens_cap: int = 512,
):
    """
    Build the prompt from real data, run generation on LLaMA 2, 
    and return an output that is capped by `max_new_tokens_cap`.

    Args:
        input_text (str): Raw text or data for the prompt builder.
        model_dir (str): Path or HF repo for your LLaMA2 model.
        min_new_tokens (int): Ensures we generate at least this many tokens.
        max_new_tokens_cap (int): Hard upper bound on new tokens generated.

    Returns:
        str: The final generated text (prompt echo stripped).
    """
    # 1) Load model & tokenizer if needed
    # if (model is None and tokenizer is None):
    #     model, tokenizer = load_LLAMA2_model(model_dir)

    # 2) Build the final prompt from the input data
    prompt = input_to_prompt_converter.build_llama2_prompt(input_text)
    # logger.info("Prompt is ready. Calculating lengths and generating...")

    # 3) Measure the prompt and input length in tokens
    prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    # 4) Move everything to the model's device
    device = model.device
    for k, v in prompt_tokens.items():
        prompt_tokens[k] = v.to(device)

    # 5) Create generation config
    gen_config = GenerationConfig(
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,  # helps reduce repeated tokens
        # Additional parameters (e.g., num_beams) if desired
    )

    # 6) Generate
    with torch.no_grad():
        outputs = model.generate(**prompt_tokens, generation_config=gen_config)

    # 7) Decode
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # 8) LLaMA often re-echos the prompt. Remove it if present:
    result = decoded[0]
    if result.startswith(prompt):
        result = result[len(prompt):]

    # 9) Remove "Summary: " prefix if present
    result = result.lstrip()
    summary_prefix = "Summary:"
    if result.startswith(summary_prefix):
        result = result[len(summary_prefix):]

    return result.strip()


if __name__ == "__main__":
    """
    Example usage: 
    - load model, build prompt from some dummy data, generate, print result.
    """
    import dfs_to_input_converter

    dummy_input_data = dfs_to_input_converter.generate_dummy_input_text()

    pred = predict(input_text=dummy_input_data, model_dir="model/llama2/")
    print("\n---------- LLaMA 2 PREDICTION ----------\n")
    print(pred)
