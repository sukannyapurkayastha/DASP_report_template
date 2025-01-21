# preprocessing.py

import json
import os
import random
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

                data.append({'input': input_text, 'output': output_text})
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: JSON decode error: {e}. Skipping.")
                continue  # Skip malformed lines
    logger.info(f"Loaded {len(data)} valid data points.")
    return data

def split_data(
    data: List[Dict[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Shuffle and split data into training, validation, and test sets.

    Args:
        data (List[Dict[str, str]]): The complete dataset.
        train_ratio (float): Proportion of data to include in the training set.
        val_ratio (float): Proportion of data to include in the validation set.
        test_ratio (float): Proportion of data to include in the test set.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
            The training, validation, and test datasets.
    """
    logger.info("Shuffling and splitting data")
    random.seed(seed)
    random.shuffle(data)

    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    logger.info(f"Training set: {len(train_data)} samples")
    logger.info(f"Validation set: {len(val_data)} samples")
    logger.info(f"Test set: {len(test_data)} samples")

    return train_data, val_data, test_data

def save_jsonl(data: List[Dict[str, str]], file_path: str):
    """
    Save a list of dictionaries to a JSON Lines file.

    Args:
        data (List[Dict[str, str]]): The dataset to save.
        file_path (str): Path to the output JSONL file.
    """
    logger.info(f"Saving {len(data)} samples to {file_path}")
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')
    logger.info(f"Saved data to {file_path}")

def get_data(
    file_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Load, clean, and split the dataset.

    Args:
        file_path (str): Path to the input JSONL file.
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        test_ratio (float): Ratio of test data.
        seed (int): Random seed for shuffling.

    Returns:
        Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
            Training, validation, and test datasets.
    """
    data = load_jsonl(file_path)

    if not data:
        logger.error("No valid data to process.")
        return [], [], []

    train_data, val_data, test_data = split_data(
        data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    return train_data, val_data, test_data

def main():
    """
    When run as a script, preprocess the data and save train, val, test splits as JSONL files.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess JSONL data for LLM training.")
    parser.add_argument('--input', type=str, default="data/real_world_data_labeled.jsonl", help='Path to the input JSONL file.')
    parser.add_argument('--output_dir', type=str, default = "data", help='Directory to save the processed data.')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of training data.')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of validation data.')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of test data.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data shuffling.')

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        logger.error("The sum of train_ratio, val_ratio, and test_ratio must be 1.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    train_data, val_data, test_data = get_data(
        file_path=args.input,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    if not train_data or not val_data or not test_data:
        logger.error("One or more data splits are empty. Check your input data and ratios.")
        return

    # Save splits
    save_jsonl(train_data, os.path.join(args.output_dir, 'train.jsonl'))
    save_jsonl(val_data, os.path.join(args.output_dir, 'val.jsonl'))
    save_jsonl(test_data, os.path.join(args.output_dir, 'test.jsonl'))

    logger.info("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()
