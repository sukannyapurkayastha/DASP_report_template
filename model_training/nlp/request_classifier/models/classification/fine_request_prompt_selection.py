import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -----------------------------
# BEISPIEL: Verschiedene Few-Shot-Versionen
# -----------------------------
few_shot_examples_v1 = {
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
        "- The 3rd line of Lemma 1: epsilon1 should be epsilon\\_1.",
        "- Page 14, Eq(14), \( \\lambda \) should be s.",
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


few_shot_examples_v2 = {
    "Request for Improvement": [
        "I believe that the presentation of the proposed method can be significantly improved.",
        "Most of my comments are improvements which can be easily included.",
        "I would replace these values with N/A or something similar.",
        "I think that the function F has to be differentiable, and this should be mentioned in the text.",
        "For instance, Eq. 2,3 can be easily combined using the proportional symbol, Eq. 8,9,10,11 show actually the same thing.",
    ],
    "Request for Explanation": [
        "How are the lambda and threshold parameters tuned? The authors mention a validation set, are they just exhaustively explored on a 3D grid on the validation set?",
        "Is there any explanation for this?",
        "The results only compare with Shim et al. Why only this method? Why would it be expected to be faster than all the other alternatives? Wouldn't similar alternatives like the sparsely gated MoE, D-softmax and adaptive-softmax have chances of being faster?",
        "4) Figure 3: Are the results averaged over multiple independent runs? If so, how many runs did you perform and could you also report confidence intervals? Since all methods are close to each other, it is hard to estimate how significant the difference is.",
        "3. In the experiments, there are large discrepancies between different optimizers on Cakewalk (e.g. SGA vs AdaGrad, Table 4). Is there any explanation for this?",
    ],
    "Request for Experiment": [
        "For example, in Sections 4 and 5 I was hoping to see comparisons to methods like MAML.",
        "Also, this work would benefit significantly from a better experimental evaluation.",
        "Another possible extension is to test this larger set of words on a non-behavioral NLP task to show possible improvements.",
        "Have the authors compared the performances of their work and [Z Hu, arXiv:1905.13728] using the same data?",
        "1) Besides an comparison to the work by Lakshminarayanan et. al, I would also like to have seen a comparison to other existing Bayesian neural network approaches such as stochastic gradient Markov-Chain Monte-Carlo methods.",
    ],
    "Request for Typo Fix": [
        "There are some typos that can be easily found, such as 'of the out algorithm'.",
        "- Capitalize: “section” -> “Section”, “appendix” -> “Appendix”, “fig.” -> “Figure”.",
        "- “Hold-out” vs “held-out”",
        "- Pg. 5, Section 3.4: '...this is would achieve...'",
        "- Pg. 6: ...thedse value of 90...",
    ],
    "Request for Clarification": [
        "These numbers correspond to several images, or to a unique image?",
        "Also, how do you select the number of factors of each type?",
        "4. How were the hyperparameters (learning rate, AdaGrad δ, Adam β1, β2) chosen?",
        "Are the curvatures the same for each layer for the GCNs?",
        "That makes sense -- but at what cost?",
    ],
    "Request for Result": [
        "Most importantly, I would like to see a measure of variance or uncertainty like confidence intervals included in the results.",
        "I would have been interested in 'false detection' experiments: comparing estimators where the mutual information is zero.",
        "It would be helpful to present the confusion matrix in your results.",
        "Additional results on the performance under varying conditions would be useful.",
        "Please report the standard deviations along with the mean values in Table 2.",
    ],
}

few_shot_examples_v3 = {
    "Request for Improvement": [
        "I suggest reorganizing the second section to highlight the main contributions more clearly.",
        "Could you refine the notation in your proof so that it’s easier to follow?",
        "It might help to restructure the experiment section into distinct subsections for clarity and flow.",
        "The paper would benefit from a more concise summary of the results in the conclusion.",
        "I recommend breaking the method description into smaller steps to improve its readability.",
    ],
    "Request for Explanation": [
        "Could you provide more insight into how you selected the hyperparameters for your model?",
        "Can you explain the reasoning behind using an L1 penalty in this context instead of L2?",
        "I’d like more details on why the proposed approach outperforms existing methods on larger datasets.",
        "What is the rationale for adopting a Bayesian prior in your model rather than a frequentist approach?",
        "Could you elaborate on the discrepancy between the theoretical results and the empirical findings?",
    ],
    "Request for Experiment": [
        "It would be helpful to see a comparison with additional baselines under similar conditions.",
        "I recommend including an ablation study to see how each component of your method contributes to the final performance.",
        "Have you tested your approach on more realistic or larger-scale datasets to validate generalizability?",
        "Could you run sensitivity analyses on key hyperparameters to understand their impact on performance?",
        "I would appreciate additional experiments that compare running times across different architectures.",
    ],
    "Request for Typo Fix": [
        "On page 5, 'labled' should be corrected to 'labeled' in the third paragraph.",
        "Please fix the reference to 'resluts' in Section 2.2— it should read 'results'.",
        "In the sentence 'we define a genral operator,' the word 'genral' should be 'general'.",
        "Could you correct 'tradiitonal methods' to 'traditional methods' in your introduction?",
        "In Equation (7), there seems to be a missing parenthesis that needs to be inserted.",
    ],
    "Request for Clarification": [
        "Could you clarify how you handle missing data points during training?",
        "I'm unsure about the difference between 'local embeddings' and 'global embeddings' in your approach—can you elaborate?",
        "How are the threshold parameters chosen for each layer, and are they tuned differently for each dataset?",
        "Could you explain how the algorithm deals with unseen labels at inference time?",
        "I’m not entirely clear on how the model’s attention mechanism weights different parts of the input—could you provide details?",
    ],
    "Request for Result": [
        "It would be useful to provide the standard deviations along with the mean error rates in Table 2.",
        "Could you include a confusion matrix to illustrate the model’s performance across the different classes?",
        "Please show the confidence intervals so we can assess statistical significance of the improvements.",
        "I’d like to see a separate column reporting runtime and memory usage for each experimental setting.",
        "Could you provide a clearer breakdown of precision and recall scores for each sub-category?",
    ],
}
# Du kannst weitere Versionen hier anlegen, z. B. few_shot_examples_v2, few_shot_examples_v3, usw.
# Für das Beispiel bleibt es bei einer Version. Du kannst sie selbst erweitern.

# Diese Variable wählst du später aus, um die "Version" der Few-Shot-Beispiele zu bestimmen.



# -----------------------------
# Label- und Kategorie-Mappings
# -----------------------------
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
    """
    Erzeugt den Prompt auf Basis der übergebenen few_shot_examples 
    (die bei Bedarf aus v1, v2, v3 ... stammen können).
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

    # Beispielhafter Aufbau: Nimm aus jeder Kategorie den ersten Satz und ergänze ihn um "Reasoning: ..."
    for label, sentences in few_shot_examples.items():
        number = label_map[label] + 1
        reasoning = "Reasoning: " + " ".join(sentences[0].split()[:10]) + "..."
        prompt += f"Sentence: \"{sentences[0]}\"\n{reasoning}\nCategory Number: {number}\n\n"

    # Nun folgt die eigentliche Anfrage
    prompt += f"Sentence: \"{query}\"\nReasoning:"
    return prompt


def map_prediction_to_label(pred, label_map):
    """
    Versucht den generierten Text (pred) mit einem Eintrag im label_map zu matchen.
    """
    pred = pred.strip().lower()
    pred = pred.strip('."\'<>/ ').lower()

    # Direkter Vergleich mit Labelnamen
    for label in label_map:
        if pred == label.lower():
            return label_map[label]

    # Falls der generierte Text das Label als Substring enthält
    for label in label_map:
        if label.lower() in pred:
            return label_map[label]

    # Falls der generierte Text nur die Nummer enthält
    for idx, label in enumerate(label_map.keys(), 1):
        if pred == str(idx) or pred == f"{idx}.":
            return label_map[label]

    # Falls keines gepasst hat, gib -1 zurück
    return -1  


def generate_predictions_from_dataset(dataset, few_shot_examples, tokenizer, model, max_new_tokens=50):
    """
    Erzeugt die Modellvorhersagen auf Basis der in few_shot_examples hinterlegten Beispiele.
    """
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


def evaluate_model(dataset, predictions, label_map):
    """
    Berechnet Accuracy, F1, Precision, Recall und zeigt die Confusion Matrix.
    """
    true_labels = dataset['fine_review_action'].map(lambda x: label_map[fine_to_category_map[x]]).tolist()
    predicted_labels = [map_prediction_to_label(pred, label_map) for pred in predictions]
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=list(label_map.values()))

    return accuracy, f1, precision, recall, cm
    


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Few-shot example versions
    # -------------------------------------------------------------------------
    few_shot_examples_dict = {
        "v1": few_shot_examples_v1,
        "v2": few_shot_examples_v2,
        "v3": few_shot_examples_v3,
    }

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
        #save_model_and_tokenizer(model, tokenizer, save_directory)

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # -----------------------------
    # 1) Modell und Tokenizer laden
    # -----------------------------
    model_name = "google/flan-t5-xl"
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # --------------------------------------
    # 2) Testdaten laden (Pfad ggf. anpassen)
    # --------------------------------------
    test_file = "backend/nlp/request_classifier/DISAPERE/final_dataset/fine_request/dev.csv"
    test_data = pd.read_csv(test_file)
    output_file = "evaluation_metrics.txt"
    # -------------------------------------
    # 3) Vorhersagen generieren (Few-Shot)
    # -------------------------------------
    print("\n--- Generating Predictions for Test Dataset ---")
    # Hier kannst du auswählen, welche Few-Shot-Version genutzt werden soll:
    for version, few_shot_examples in few_shot_examples_dict.items():
        predictions = generate_predictions_from_dataset(test_data, few_shot_examples, tokenizer, model)
        metrics = evaluate_model(test_data, predictions, label_map)
       
        with open(output_file, "a", encoding="utf-8") as f:  # Use "a" to append instead of overwriting
            f.write(f"Results for version: {version}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write("Confusion Matrix:\n")
            for row in metrics['confusion_matrix']:
                f.write("\t".join(map(str, row)) + "\n")
            f.write("\n\n")
        
    
    
