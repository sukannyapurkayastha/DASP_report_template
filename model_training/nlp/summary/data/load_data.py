import json
import os
from tqdm import tqdm
import logging
from typing import List, Dict
import input_to_prompt_converter

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_jsonl(file_path: str):
    """
    Validate each line in a JSONL file.

    Args:
        file_path (str): Path to the JSONL file.
    """
    with open(file_path, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                assert 'input' in data and 'output' in data, f"Missing fields in line {i}"
            except (json.JSONDecodeError, AssertionError) as e:
                logging.error(f"Invalid JSON or missing fields on line {i}: {e}")
                raise e


def load_jsonl(file_path: str) -> List[Dict[str, str]]:
    """
    Load a JSON Lines file into a list of dictionaries.
    Each line should contain 'input' and 'output' keys.
    Ensures that 'input' and 'output' are non-empty strings.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        List[Dict[str, str]]: List of data points with 'input' and 'output'.
    """
    data = []
    logger.info(f"Loading data from {file_path}")
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        return data

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading JSONL"), 1):
            try:
                json_line = json.loads(line)
                input_text = json_line.get('input', "")
                output_text = json_line.get('output', "")

                # Ensure 'input' is a string
                if not isinstance(input_text, str):
                    input_text = str(input_text) if input_text is not None else ""

                # Ensure 'output' is a string
                if not isinstance(output_text, str):
                    output_text = str(output_text) if output_text is not None else ""

                # Strip whitespace and check for non-empty strings
                input_text = input_text.strip()
                output_text = output_text.strip()

                if not input_text or not output_text:
                    logger.warning(f"Line {line_num}: Empty 'input' or 'output'. Skipping.")
                    continue

                # convert input into a prompt
                input_text = input_to_prompt_converter.build_prompt(input_text)
                data.append({'input': input_text, 'output': output_text})
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: JSON decode error: {e}. Skipping.")
                continue  # Skip malformed lines
    logger.info(f"Loaded {len(data)} valid data points.")
    return data

def main():
    # Validate training and validation data
    validate_jsonl('data/train.jsonl')
    validate_jsonl('data/val.jsonl')
main()
