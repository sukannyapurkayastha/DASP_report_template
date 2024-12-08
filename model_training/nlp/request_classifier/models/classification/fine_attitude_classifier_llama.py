import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

few_shot_examples = {
    "Clarity": [
        "With regard to my first negative point above about the lack of discussions, it seems the analysis of Section 4 is disproportionate compared to other places.",
        "- My main concern about the analysis is that it shows why several methods (e.g., momentum, multiple update steps) are *not* helpful for stabilising GANs, but does not tell why training with these methods, as well as others such as gradient penalty, *do converge* in practice with properly chosen hyper-parameters?",
        "The details of the approach is not entirely clear and no theoretical results are provided to support the approach.",
    ],
    "Meaningful-comparison": [
        "- I’d also like to see more extensive comparisons between FICM and ICM across different datasets, for example, Super Mario Bros. and the Atari games, instead of only comparing FICM against ICM on ViZDoom.",
        "2: The authors should compare against several costs/algorithms (e.g. l_0 with OMP, l_1 with LARS, etc.), and across various N_0/sparsity penalties, and across several datasets.",
        "The authors also rely mostly on the FID metric, but do not show if and how there is improvement upon visual inspection of the generated images (i.e., is resolution improved, is fraction of images that look clearly 'unnatural' reduced etc.)",
    ],
    "Motivation-impact": [
        "If faster training of dictionary learning models was a bottleneck in practical applications, this might be of interest, but it is not.",
        "It is also not clear to me why these problems are important.",
        "No baseline comparison with GraphNets.",
    ],
    "Orginality": [
        "Overall, the method looks incremental and experimental results are mixed on small datasets so I vote for rejection.",
        "That (except the minor small section of streaming data), the paper is more like a proper verification of how tree-based learning algorithms work very well in tabular data--which is far from the basis of the paper and does not make the paper novel enough for ICLR.",
        "My main concern comes from the novelty of this paper.",
    ],
    "Replicability": [
        "The authors have not provided enough details for reproducing the experiments.",
        "Without the code and data, it's difficult to verify the claims.",
        "Some crucial implementation details are missing, making it hard to replicate the results.",
    ],
    "Soundness-correctness": [
        "The proof of Theorem 2 seems incorrect under the given assumptions.",
        "There are some logical inconsistencies in the argument presented in Section 3.",
        "The algorithm does not converge as claimed due to the error in the update rule.",
    ],
    "Substance": [
        "The paper lacks depth in its analysis and doesn't contribute significantly to the field.",
        "There is not enough evidence to support the main claims.",
        "The work seems preliminary and needs more substantial results.",
    ],
    "Other": [
    "The paper uses standard methodologies without introducing new ideas.",
    "General comments about the paper’s overall quality or readability.",
    "The authors make minor contributions that are unrelated to the main discussion."
    ],
}

def create_few_shot_prompt(query, few_shot_examples):
    prompt = (
        "As an assistant, analyze the sentence and determine its category based on the reasoning.\n"
        "Select the best category from the following options. Use 'Other' only if none of the other categories apply.\n"
    )
    for label in label_map:
        prompt += f"- {label}\n"
    prompt += "\n"

    for label, sentences in few_shot_examples.items():
        reasoning = "Reasoning: " + " ".join(sentences[0].split()[:10]) + "..."
        prompt += f"Sentence: \"{sentences[0]}\"\n{reasoning}\nCategory: {label}\n\n"
    prompt += f"Sentence: \"{query}\"\nReasoning:"
    return prompt



def map_prediction_to_label(pred, label_map):
    # Clean the prediction text
    pred = pred.replace('<pad>', '').strip().lower()
    
    # Try to extract the category from the prediction
    import re
    match = re.search(r'category[:\- ]*(.*)', pred, re.IGNORECASE)
    if match:
        pred_label = match.group(1).strip()
        # Direct match
        for label in label_map:
            if pred_label.lower() == label.lower():
                return label_map[label]
        # Partial match
        for label in label_map:
            if label.lower() in pred_label.lower():
                return label_map[label]
    
    # Fallback to check the entire prediction
    for label in label_map:
        if label.lower() in pred:
            return label_map[label]
    
    return -1  # Return -1 if no match found


def generate_predictions_from_dataset(dataset, few_shot_examples, tokenizer, model, max_new_tokens=50, temperature=0.3):
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
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(generated_text.strip())
    return predictions

def evaluate_model(dataset, predictions, label_map):
    # Map true labels to numerical labels using label_map
    print(dataset)
    true_labels = dataset['aspect'].map(lambda x: label_map[fine_to_category_map[x]]).tolist()
    # Map predicted labels to numerical labels
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

    test_file = "backend/nlp/request_classifier/DISAPERE/final_dataset/fine_request/test_attitude.csv"  # Update with your test dataset path
    test_data = pd.read_csv(test_file)
    test_data = test_data[test_data["aspect"] != "none"]
    test_data = test_data.reset_index(drop=True)


    # Define label map
    label_map = {
        "Other": 0,
        "Clarity": 1,
        "Meaningful-comparison": 2,
        "Motivation-impact": 3,
        "Originality": 4,  # Fixed typo from "Orginality"
        "Replicability": 5,
        "Soundness-correctness": 6,
        "Substance": 7,
    }


    fine_to_category_map = {
    "arg_other": "Other",
    "asp_clarity": "Clarity",
    "asp_meaningful-comparison": "Meaningful-comparison",
    "asp_motivation-impact": "Motivation-impact",
    "asp_originality": "Originality",  # Fixed typo from "asp_orginality"
    "asp_replicability": "Replicability",
    "asp_soundness-correctness": "Soundness-correctness",
    "asp_substance": "Substance",
    }



    print("\n--- Generating Predictions for Test Dataset ---")
    predictions = generate_predictions_from_dataset(test_data, few_shot_examples, tokenizer, model)

    for i, pred in enumerate(predictions[:10]):  
        print(f"Input: {test_data['text'][i]}")
        print(f"Predicted Output: {pred}")
        print("---")

    print("\n--- Evaluating Model on Test Dataset ---")
    evaluate_model(test_data, predictions, label_map)
