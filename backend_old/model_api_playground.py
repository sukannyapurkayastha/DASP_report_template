from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoConfig
import torch

# Initialize FastAPI
app = FastAPI()

# Load model and tokenizer
MODEL_NAME = "/home/nana/EMNLP2023_jiu_jitsu_argumentation_for_rebuttals/codes/review_to_desc/t5-large-output/3/1e-4/"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Load the model configuration
config = AutoConfig.from_pretrained(MODEL_NAME)
# Access the task-specific parameters for summarization
summarization_params = config.task_specific_params["summarization"]

# Define input schema
class InputData(BaseModel):
    text: str

@app.post("/predict/")
async def predict(data: InputData):
    inputs = tokenizer(data.text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        # Generate the summary using the parameters from the config
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=summarization_params["num_beams"],          # Beam search
            max_length=summarization_params["max_length"],        # Max length of the summary
            min_length=summarization_params["min_length"],        # Min length of the summary
            early_stopping=summarization_params["early_stopping"],# Early stopping when EOS token is generated
            length_penalty=summarization_params["length_penalty"],# Adjust summary length
            no_repeat_ngram_size=summarization_params["no_repeat_ngram_size"], # Avoid repetition
        )
    # Decode the generated tokens to get the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {
        "text": data.text,
        "summary": summary
    }

