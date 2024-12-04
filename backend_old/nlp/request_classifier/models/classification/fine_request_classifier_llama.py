
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

    for label, sentences in few_shot_examples.items():
        number = label_map[label] + 1
        reasoning = "Reasoning: " + " ".join(sentences[0].split()[:10]) + "..."
        prompt += f"Sentence: \"{sentences[0]}\"\n{reasoning}\nCategory Number: {number}\n\n"

    prompt += f"Sentence: \"{query}\"\nReasoning:"
    return prompt


def map_prediction_to_label(pred, label_map):
    pred = pred.strip().lower()
    pred = pred.strip('."\'<>/ ').lower()

    for label in label_map:
        if pred == label.lower():
            return label_map[label]

    for label in label_map:
        if label.lower() in pred:
            return label_map[label]


    for idx, label in enumerate(label_map.keys(), 1):
        if pred == str(idx) or pred == f"{idx}.":
            return label_map[label]

    return -1  


def generate_predictions_from_dataset(dataset, few_shot_examples, tokenizer, model, max_new_tokens=50):
    predictions = []
    for query in tqdm(dataset["text"], desc="Generating predictions"):
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
    
   
    cm = confusion_matrix(true_labels, predicted_labels, labels=list(label_map.values()))
    display_labels = list(label_map.keys())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(xticks_rotation='vertical', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
  
    model_name = "google/flan-t5-xl"
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer.pad_token_id = tokenizer.eos_token_id


    test_file = "backend/nlp/request_classifier/DISAPERE/final_dataset/fine_request/dev.csv"
    test_data = pd.read_csv(test_file)


    print("\n--- Generating Predictions for Test Dataset ---")
    predictions = generate_predictions_from_dataset(test_data, few_shot_examples, tokenizer, model)


    for i, pred in enumerate(predictions[:10]):
        print(f"Input: {test_data['text'][i]}")
        print(f"Predicted Output: {pred}")
        print("---")

    print("\n--- Evaluating Model on Test Dataset ---")
    evaluate_model(test_data, predictions, label_map)
