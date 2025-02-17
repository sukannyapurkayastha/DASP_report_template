from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load the pretrained model and tokenizer
local_path = "models/description_generator/"
huggingface_model_path = "DASP-ROG/ThemeModel"
tokenizer = T5Tokenizer.from_pretrained(huggingface_model_path, cache_dir=local_path)
model = T5ForConditionalGeneration.from_pretrained(huggingface_model_path, cache_dir=local_path, torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Example input
input_texts = ['The reason why other methods are much better than LSTNet under the setting of I/I should be clarified.',
 '4. For the SO(3)-equivariant and -invariant methods, some works for point cloud registration [2, 3, 4, 5] should also be discussed.']

# Tokenize inputs (batch size = len(input_texts))
inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
input_ids = inputs.input_ids.to(device)

# Generate outputs, with params from config
with torch.no_grad():
    output_ids = model.generate(input_ids, max_length=200, num_beams=4, length_penalty=2.0)

# Decode output
output_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in output_ids]
for text in output_texts:
    print("Generated Summary:", text)
