from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize FastAPI
app = FastAPI()

# Load model and tokenizer
MODEL_NAME = "./path/to/your/model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Define input schema
class InputData(BaseModel):
    text: str

@app.post("/predict/")
async def predict(data: InputData):
    inputs = tokenizer(data.text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return {
        "text": data.text,
        "predicted_label": predicted_label,
        "confidence_scores": outputs.logits.softmax(dim=1).tolist()
    }

# Run the server with: uvicorn app:app --reload
