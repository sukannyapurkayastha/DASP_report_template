import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Define few-shot examples and label maps
few_shot_examples = {
    "Request for Improvement": [
        "Most of my comments are improvements which can be easily included.",
        "These changes are minor and should be simple to implement.",
        "I believe the suggested enhancements can be quickly added.",
        "My comments are straightforward improvements that can be easily made.",
        "The improvements I suggest can be easily incorporated.",
    ],
    "Request for Explanation": [
        "Is there any explanation for this?",
        "Could you provide more details on how the loss function is derived?",
        "Please explain why we need to add MC-Dropout to the ensemble.",
        "What is the benefit of your method over existing approaches?",
        "However, I do not understand how the discrete output y is handled.",
    ],
    "Request for Experiment": [
        "It would be nice if more network architectures were analyzed (such as VGG and DenseNets).",
        "Also, this work would benefit significantly from a better experimental evaluation.",
        "For example, in Sections 4 and 5 I was hoping to see comparisons to methods like MAML.",
        "I would suggest including empirical evidence in the experiments to show convergence.",
        "Another possible extension is to test this larger set of words on a non-behavioral NLP task to show possible improvements.",
    ],
    "Request for Typo Fix": [
        "- 'principle curvatures' should be 'principal curvatures'.",
        "- Page 2: 'network's type to be class' should be 'to be a class'.",
        "- The end of the 2nd line of Lemma 1: P, G should be \\( \\mathbb{P}, \\mathbb{G} \\).",
        "- The 3rd line of Lemma 1: epsilon1 should be epsilon\_1.",
        "- Page 14, Eq(14), \( \lambda \) should be s.",
    ],
    "Request for Clarification": [
        "Can you clarify how you view the relationship between the approaches mentioned above?",
        "Also, how do you select the number of factors of each type?",
        "These numbers correspond to several images, or to a unique image?",
        "I don’t get the details of the batch normalization used: with respect to which axis are the mean and variance computed?",
        "This parameter Omega is estimated individually for each degraded image?",
    ],
    "Request for Result": [
        "Most importantly, I would like to see a measure of variance or uncertainty like confidence intervals included in the results.",
        "I would have been interested in 'false detection' experiments: comparing estimators where the mutual information is zero.",
        "It would be helpful to present the confusion matrix in your results.",
        "Additional results on the performance under varying conditions would be useful.",
        "Please report the standard deviations along with the mean values in Table 2.",
    ],
}

label_map = {
    "Request for Improvement": 0,
    "Request for Explanation": 1,
    "Request for Experiment": 2,
    "Request for Typo Fix": 3,
    "Request for Clarification": 4,
    "Request for Result": 5,
    "Request unclear": -1,
}

fine_to_category_map = {
    "arg-request_edit": "Request for Improvement",
    "arg-request_explanation": "Request for Explanation",
    "arg-request_experiment": "Request for Experiment",
    "arg-request_typo": "Request for Typo Fix",
    "arg-request_clarification": "Request for Clarification",
    "arg-request_result": "Request for Result",
}


def create_few_shot_prompt(query, few_shot_examples):
    """
    Creates a few-shot learning prompt for the T5 model.

    Parameters:
    query (str): The input sentence to classify.
    few_shot_examples (dict): Dictionary containing few-shot examples for each category.

    Returns:
    str: The generated few-shot learning prompt.
    """
    prompt = (
        "As an assistant, analyze the sentence and determine its category based on the reasoning.\n"
        "Select the area which is the subject or the type of request.\n\n"
        "Categories:\n"
        "Request for Improvement\n"
        "Request for Explanation\n"
        "Request for Experiment\n"
        "Request for Typo Fix\n"
        "Request for Clarification\n"
        "Request for Result\n\n"
    )

    # Add examples to the prompt
    for label, sentences in few_shot_examples.items():
        number = label_map[label] + 1
        reasoning = "Reasoning: " + " ".join(sentences[0].split()[:10]) + "..."
        prompt += f"Sentence: \"{sentences[0]}\"\n{reasoning}\nCategory Number: {number}\n\n"

    # Add the query sentence
    prompt += f"Sentence: \"{query}\"\nReasoning:"
    return prompt


def map_prediction_to_label(pred, label_map):
    """
    Maps model prediction to a corresponding label in the label map.

    Parameters:
    pred (str): The prediction text generated by the model.
    label_map (dict): Dictionary mapping categories to their label indices.

    Returns:
    int: Mapped label index or -1 if not found.
    """
    pred = pred.strip().lower()
    pred = pred.strip('."\'<>/ ').lower()

    # Exact match with label
    for label in label_map:
        if pred == label.lower():
            return label_map[label]

    # Partial match
    for label in label_map:
        if label.lower() in pred:
            return label_map[label]

    # Match with numeric category
    for idx, label in enumerate(label_map.keys(), 1):
        if pred == str(idx) or pred == f"{idx}.":
            return label_map[label]

    return -1  # Default for unmatched predictions


def generate_predictions_from_dataset(dataset, few_shot_examples, tokenizer, model, max_new_tokens=50):
    """
    Generates predictions for each sentence in the dataset using few-shot prompts.

    Parameters:
    dataset (Dataset): Dataset containing sentences to classify.
    few_shot_examples (dict): Dictionary of few-shot examples.
    tokenizer (T5Tokenizer): Tokenizer for the T5 model.
    model (T5ForConditionalGeneration): Pretrained T5 model.
    max_new_tokens (int): Maximum tokens to generate for each prediction.

    Returns:
    list: List of generated predictions.
    """
    predictions = []
    for query in tqdm(dataset["sentence"], desc="Generating predictions"):
        few_shot_prompt = create_few_shot_prompt(query, few_shot_examples)
        inputs = tokenizer(
            few_shot_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(generated_text.strip())
    return predictions


def evaluate_model(dataset, predictions, label_map):
    """
    Evaluates the model's predictions against true labels and displays results.

    Parameters:
    dataset (Dataset): Dataset containing true labels.
    predictions (list): List of predicted labels.
    label_map (dict): Mapping of categories to label indices.

    Returns:
    None
    """
    true_labels = dataset['fine_review_action'].map(lambda x: label_map[fine_to_category_map[x]]).tolist()
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

    # Display confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=list(label_map.values()))
    display_labels = list(label_map.keys())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(xticks_rotation='vertical', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()


def save_model_and_tokenizer(model, tokenizer, save_directory):
    """
    Saves the model and tokenizer to the specified directory.

    Parameters:
    model (T5ForConditionalGeneration): Trained T5 model.
    tokenizer (T5Tokenizer): Tokenizer for the T5 model.
    save_directory (str): Directory to save the model and tokenizer.

    Returns:
    None
    """
    print(f"Saving model and tokenizer in {save_directory}...")
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print("Model and tokenizer saved.")


def load_model_and_tokenizer(save_directory):
    """
    Loads a saved model and tokenizer from the specified directory.

    Parameters:
    save_directory (str): Directory from where the model and tokenizer will be loaded.

    Returns:
    tuple: Loaded model and tokenizer.
    """
    print(f"Loading model and tokenizer from {save_directory}...")
    tokenizer = T5Tokenizer.from_pretrained(save_directory)
    model = T5ForConditionalGeneration.from_pretrained(save_directory)
    print("Loading successful.")
    return model, tokenizer


if __name__ == "__main__":
    model_name = "google/flan-t5-xl"
    script_directory = os.path.dirname(os.path.abspath(__file__))
    save_directory = os.path.join(script_directory, "../../../../../backend/models/request_classifier/fine_request_classifier")

    try:
        model, tokenizer = load_model_and_tokenizer(save_directory)
    except Exception as e:
        print(f"Model not found or error while loading: {e}")
        print(f"Downloading model {model_name}...")
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        save_model_and_tokenizer(model, tokenizer, save_directory)

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer.pad_token_id = tokenizer.eos_token_id
