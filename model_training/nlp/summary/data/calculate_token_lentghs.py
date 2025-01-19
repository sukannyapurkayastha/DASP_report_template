import json
from transformers import T5Tokenizer
from datasets import load_dataset

# Initialize the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

def load_jsonl(file_path):
    """
    Load a JSON Lines file into a list of dictionaries.
    Each line should contain 'input' and 'output' keys.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line)
            data.append(json_line)
    return data

def calculate_token_lengths(data, tokenizer):
    input_lengths = []
    output_lengths = []
    
    for example in data:
        input_tokens = tokenizer.encode(example['input'], add_special_tokens=True)
        output_tokens = tokenizer.encode(example['output'], add_special_tokens=True)
        
        input_lengths.append(len(input_tokens))
        output_lengths.append(len(output_tokens))
    
    return input_lengths, output_lengths

def main():
    dataset_path = 'data/real_world_data_labeled.jsonl'  # Update this path if necessary
    data = load_jsonl(dataset_path)
    
    input_lengths, output_lengths = calculate_token_lengths(data, tokenizer)
    
    print(f"Input Tokens: Min={min(input_lengths)}, Max={max(input_lengths)}, Avg={sum(input_lengths)/len(input_lengths):.2f}")
    print(f"Output Tokens: Min={min(output_lengths)}, Max={max(output_lengths)}, Avg={sum(output_lengths)/len(output_lengths):.2f}")
    
    # Optional: Visualize distribution
    try:
        import matplotlib.pyplot as plt
        plt.hist(input_lengths, bins=20, alpha=0.5, label='Input Lengths')
        plt.hist(output_lengths, bins=20, alpha=0.5, label='Output Lengths')
        plt.legend(loc='upper right')
        plt.xlabel('Token Length')
        plt.ylabel('Frequency')
        plt.title('Distribution of Token Lengths')
        plt.show()
    except ImportError:
        print("matplotlib not installed. Skipping the histogram.")

if __name__ == "__main__":
    main()
