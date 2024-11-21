import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from tqdm import tqdm


few_shot_examples = {
    "Request for Improvement": [
        "I would recommend that the authors perhaps shorten section 3 or remove figure 9 to fit it into 8 pages.",
        "I found that the core technical description was quite brief and would have benefited from simply more detail and space.",
        "The contributions of the method could also be underlined more clearly in the abstract and introduction.",
    ],
    "Request for Explanation": [
        "Can you elaborate more on why BatchNorm statistics are computed across all devices as opposed to per-device? Was this crucial for best performance?",
        "Why not use continuous actions with a parameterized policy (e.g., Gaussian)?",
        "I wonder why the authors didn’t compare or mention optimizers such as ADAM and ADAGRAD which adapt their parameters on-the-fly as well.",
    ],
    "Request for Experiment": [
        "It would be nice if more network architectures were analysed (such as VGG and DenseNets).",
        "I would strongly recommend including the computational cost of each method in the evaluation section.",
        "I would like to see how these curves vary with different parameters.",
    ],
    "Request for Typo Fix": [
        "- 'data tripets' on page 2",
        "- The word in the title should be 'Convolutional', right?",
        "- 'principle curvatures' should be 'principal curvatures'.",
    ],
    "Request for Clarification": [
        "Why was this policy used as the baseline? It seems extremely basic and unlikely to truly lead to optimal performance.",
        "- What implementation of the other algorithms did you use?",
        "Can you explain the sentence 'To prevent data being added suddenly, no data was added until 5 iterations'?",
    ],
    "Request for Result": [
        "Most importantly, I would like to see a measure of variance/uncertainty like confidence intervals included in the results.",
        "Providing such analysis would be also helpful for the community.",
        "Limitations and where the proposed method brings improvement should be highlighted.",
    ],
}

def create_few_shot_prompt(query, few_shot_examples):
    """
    Create a few-shot prompt with examples and a query.
    """
    prompt = (
        "You are an assistant that classifies sentences into one of the following categories:\n"
        "1. Request for Improvement\n"
        "2. Request for Explanation\n"
        "3. Request for Experiment\n"
        "4. Request for Typo Fix\n"
        "5. Request for Clarification\n"
        "6. Request for Result\n\n"
        "For each example, provide only the category name from the list above.\n\n"
    )


    for label, sentences in few_shot_examples.items():
        for sentence in sentences:
            prompt += f"Sentence: \"{sentence}\"\nCategory: {label}\n\n"

    # Add the query sentence
    prompt += f"Now classify this sentence:\n\"{query}\"\nCategory:"
    return prompt


def map_prediction_to_label(pred, label_map):
    pred = pred.strip().lower()
   
    for label in label_map:
        if pred == label.lower():
            return label_map[label]
    # Partial match
    for label in label_map:
        if label.lower() in pred:
            return label_map[label]
   
    if pred.isdigit():
        number = int(pred)
        if 1 <= number <= len(label_map):
            label_list = list(label_map.keys())
            return label_map[label_list[number - 1]]
    return -1  


def generate_predictions_from_dataset(dataset, few_shot_examples, tokenizer, model, max_new_tokens=50, temperature=0.3):
    """
    Generate predictions for a dataset using few-shot examples.
    """
    predictions = []
    for query in tqdm(dataset["text"], desc="Generating predictions"):
       
        few_shot_prompt = create_few_shot_prompt(query, few_shot_examples)
        
      
        inputs = tokenizer(few_shot_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_length = input_ids.shape[-1]

        outputs = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        predictions.append(generated_text.strip())
    return predictions

# Evaluate the model
def evaluate_model(dataset, predictions, label_map):
    """
    Evaluate the model predictions against true labels.
    """
    true_labels = dataset["target"].tolist()  
    predicted_labels = [map_prediction_to_label(pred, label_map) for pred in predictions]

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


if __name__ == "__main__":
   
    model_name = "meta-llama/Llama-2-7b-chat-hf"  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

   
    tokenizer.pad_token_id = tokenizer.eos_token_id

   
    test_file = "backend/nlp/request_classifier/DISAPERE/final_dataset/fine_request/test_short.csv"  
    test_data = pd.read_csv(test_file) 

    # Define label map
    label_map = {
        "Request for Improvement": 0,
        "Request for Explanation": 1,
        "Request for Experiment": 2,
        "Request for Typo Fix": 3,
        "Request for Clarification": 4,
        "Request for Result": 5,
    }


    print("\n--- Generating Predictions for Test Dataset ---")
    predictions = generate_predictions_from_dataset(test_data, few_shot_examples, tokenizer, model)


    for i, pred in enumerate(predictions[:10]):  # Print first 10 predictions
        print(f"Input: {test_data['text'][i]}")
        print(f"Predicted Output: {pred}")
        print("---")


    print("\n--- Evaluating Model on Test Dataset ---")
    evaluate_model(test_data, predictions, label_map)
