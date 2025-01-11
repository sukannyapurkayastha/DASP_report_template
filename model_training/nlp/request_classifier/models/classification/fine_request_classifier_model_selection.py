import torch
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    confusion_matrix
)
from transformers import TrainerCallback, TrainerState, TrainerControl
import pandas as pd
import os
import numpy as np
import datetime

#########################
# 1) Datacleaning-Funktionen
#########################

def clean_special_characters(dataset, column="sentence"):
    """
    Removes special characters like leading/trailing quotes from the specified column in the dataset.
    """
    def clean_example(example):
        if isinstance(example[column], str):
            example[column] = example[column].strip().replace('"', '')
        return example

    return dataset.map(clean_example)

def clean_dataset(dataset):
    """
    Cleans and filters a Hugging Face Dataset:
    - Removes sentences shorter than 10 characters or with fewer than 1 whitespace.
    - Converts all text to lowercase.
    """
    def clean_sentence(example):
        sentence = example['sentence']
        if isinstance(sentence, str):
            sentence = sentence.strip().lower()
            if len(sentence) >= 10 and sentence.count(" ") >= 1:
                return sentence
        return None

    # Filter out invalid sentences
    dataset = dataset.filter(lambda example: clean_sentence(example) is not None)
    dataset = dataset.map(lambda example: {"sentence": clean_sentence(example)})
    return dataset


#########################
# 2) Metrik-Funktionen und Callback
#########################

def compute_metrics(eval_pred):
    """
    Computes evaluation metrics for a classification model.
    """
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
    }

class MetricsLogger(TrainerCallback):
    """
    Custom callback to log evaluation metrics after each epoch.
    """
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        print(f"Metrics after epoch {state.epoch}: {metrics}")


#########################
# 3) Laden und Vorverarbeiten der Daten
#########################

def load_data(do_cleaning=True):
    """
    Loads the dataset from CSV files and optionally cleans each split.
    """
    data_files = {
        "train": "model_training/nlp/request_classifier/DISAPERE/final_dataset/fine_request/train.csv",
        "validation": "model_training/nlp/request_classifier/DISAPERE/final_dataset/fine_request/test.csv",
        "test": "model_training/nlp/request_classifier/DISAPERE/final_dataset/fine_request/dev.csv",
    }

    data = load_dataset("csv", data_files=data_files)

    if do_cleaning:
        for split in ["train", "validation", "test"]:
            data[split] = clean_dataset(data[split])
            data[split] = clean_special_characters(data[split], "sentence")

    return data

def oversample_minority_class(dataset, label_col="target"):
    """
    Oversamples the minority classes in the dataset to balance the class distribution.
    """
    df = dataset.to_pandas()
    class_counts = df[label_col].value_counts()
    max_count = class_counts.max()

    balanced_dfs = []
    for cls in class_counts.index:
        df_cls = df[df[label_col] == cls]
        if len(df_cls) < max_count:
            df_cls = df_cls.sample(max_count, replace=True, random_state=42)
        balanced_dfs.append(df_cls)

    balanced_df = pd.concat(balanced_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return Dataset.from_pandas(balanced_df, preserve_index=False)

def tokenize_function(example, tokenizer):
    """
    Tokenizes text data for BERT input.
    """
    return tokenizer(
        example["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_attention_mask=True,
    )

def prepare_datasets(tokenizer, do_cleaning=True, do_oversampling=True):
    """
    Loads, optionally cleans, optionally oversamples, then tokenizes the train, validation, test datasets.

    Returns: (train_ds, val_ds, test_ds) - each with columns ["input_ids","attention_mask","labels"].
    """
    # 1) Daten laden
    data = load_data(do_cleaning=do_cleaning)

    # 2) Oversampling nur auf Training
    if do_oversampling:
        train_dataset = oversample_minority_class(data["train"], label_col="target")
    else:
        train_dataset = data["train"]

    val_dataset = data["validation"]
    test_dataset = data["test"]

    # 3) Tokenisieren
    def map_fn(x): 
        return tokenize_function(x, tokenizer)
    
    tokenized_train = train_dataset.map(map_fn, batched=True)
    tokenized_val = val_dataset.map(map_fn, batched=True)
    tokenized_test = test_dataset.map(map_fn, batched=True)

    # 4) Labels-Spalte umbenennen
    tokenized_train = tokenized_train.rename_column("target", "labels")
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    tokenized_val = tokenized_val.rename_column("target", "labels")
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    tokenized_test = tokenized_test.rename_column("target", "labels")
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized_train, tokenized_val, tokenized_test


#########################
# 4) Thresholding Logic from Second Snippet
#########################
### NEW/UPDATED ###
def apply_thresholds(logits, class_thresholds):
    """
    Applies class-specific thresholds to predictions.
    For each example:
      1) Convert logits -> softmax probabilities.
      2) For each class, check if p >= class_thresholds[class_idx].
         Collect all "valid" classes.
      3) If none are valid, pick the argmax class.
      4) If >=1 is valid, pick the one with the highest prob among the valid ones.
    """
    probabilities = torch.softmax(torch.tensor(logits), dim=-1)
    predictions = []

    for prob in probabilities:
        valid_classes = []
        for class_idx, p in enumerate(prob):
            # Default threshold is 0.5 if not in dictionary
            if p >= class_thresholds.get(class_idx, 0.5):
                valid_classes.append((class_idx, p))

        if valid_classes:
            predicted_class = max(valid_classes, key=lambda x: x[1])[0]
        else:
            predicted_class = torch.argmax(prob).item()

        predictions.append(predicted_class)
    return predictions


#########################
# 5) Haupt-Training & Vorhersage
#########################

### NEW/UPDATED ###
def train_and_evaluate_model(
    model_name,
    do_cleaning=False,
    do_oversampling=False,
    num_epochs=5,
    batch_size=16,
    results_file_path="metrics_results.txt",
    do_thresholding=False,              # <--- new param
    class_thresholds=None,              # <--- new param
    num_labels=6                        # <--- changed default from 2 to 6
):
    """
    Trains a BERT-based model for sequence classification:
    - model_name can be a HF Hub model (e.g. "bert-base-uncased") or a local path (e.g. "classification/custom_model").
    - do_cleaning/do_oversampling toggles for data cleaning and oversampling.
    - do_thresholding applies class-specific thresholds if True.
    - class_thresholds is a dict like {0: 0.35, 1: 0.35, 2: 0.25, 3: 0.20, ...}
    - num_labels: number of output classes (defaults to 6).
    """
    # 1) Tokenizer/Model laden
    # Unterscheidung: Lokaler Pfad vs. Hugging-Face-Hub
    if model_name.startswith("model_training/"):
        tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(model_name, local_files_only=True, num_labels=num_labels)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # 2) Datensätze vorbereiten
    train_ds, val_ds, test_ds = prepare_datasets(
        tokenizer,
        do_cleaning=do_cleaning,
        do_oversampling=do_oversampling
    )

    # 3) TrainingArguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
    )

    # 4) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[MetricsLogger()],
    )

    # 5) Training
    trainer.train()

    # 6) Evaluate on Validation & Test
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    test_metrics = trainer.evaluate(eval_dataset=test_ds)

    # 7) Confusion Matrix (with or without thresholding)
    test_output = trainer.predict(test_ds)
    pred_logits = test_output.predictions
    true_labels = test_output.label_ids

    ### NEW/UPDATED ###
    if do_thresholding and class_thresholds is not None:
        # Use apply_thresholds from above
        pred_labels = apply_thresholds(pred_logits, class_thresholds)
        # If you want a confusion matrix over ONLY the keys in class_thresholds, do:
        # cm = confusion_matrix(true_labels, pred_labels, labels=list(class_thresholds.keys()))
        # But usually you'd do:
        cm = confusion_matrix(true_labels, pred_labels)
    else:
        # No thresholding => standard argmax
        pred_labels = np.argmax(pred_logits, axis=-1)
        cm = confusion_matrix(true_labels, pred_labels)

    # 8) Ergebnisse in Datei schreiben
    with open(results_file_path, "a", encoding="utf-8") as f:
        f.write("==============================================\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Data Cleaning: {do_cleaning}\n")
        f.write(f"Oversampling: {do_oversampling}\n")
        f.write(f"Thresholding: {do_thresholding}\n")
        f.write(f"Epochs: {num_epochs}\n\n")

        f.write("Validation Metrics:\n")
        f.write(f"  Accuracy:  {val_metrics['eval_accuracy']:.4f}\n")
        f.write(f"  Precision: {val_metrics['eval_precision']:.4f}\n")
        f.write(f"  Recall:    {val_metrics['eval_recall']:.4f}\n")
        f.write(f"  F1-Score:  {val_metrics['eval_f1']:.4f}\n")

        f.write("\nTest Metrics:\n")
        f.write(f"  Accuracy:  {test_metrics['eval_accuracy']:.4f}\n")
        f.write(f"  Precision: {test_metrics['eval_precision']:.4f}\n")
        f.write(f"  Recall:    {test_metrics['eval_recall']:.4f}\n")
        f.write(f"  F1-Score:  {test_metrics['eval_f1']:.4f}\n\n")

        f.write("Confusion Matrix (Test):\n")
        f.write(str(cm) + "\n")
        f.write("==============================================\n\n")


def predict(texts, model_path="./bert_request_classifier_model"):
    """
    Predicts labels for a list of texts using a trained BERT model.
    """
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    return torch.argmax(logits, dim=-1).cpu().numpy()


#########################
# 6) Hauptprogramm (main)
#########################
def main():
    """
    Example: trains the model on 2 model variants, with/without cleaning, with/without oversampling, 
    with/without thresholding => multiple runs.
    Adjust as needed.
    """
    results_file_path = "metrics_results.txt"
    # Start fresh results file:
    with open(results_file_path, "w", encoding="utf-8") as f:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Ergebnis-Log gestartet am {current_time}\n\n")

    # Example: Two model names
    model_names = [
        "bert-base-uncased",
        "allenai/scibert_scivocab_uncased",
        "model_training/nlp/request_classifier/models/classification/sciBERT_neg"
    ]

    ### NEW/UPDATED ###
    # Example class thresholds for 6 classes
    # Adjust the threshold values to your liking
    class_thresholds = {
        0: 0.35,
        1: 0.35,
        2: 0.25,
        3: 0.20,
        4: 0.25,
        5: 0.35
    }

    # 2 (cleaning) x 2 (oversampling) x 2 (thresholding) x 2 (models) = 16 runs
    for model_name in model_names:
        for cleaning in [False, True]:
            for oversampling in [False, True]:
                for thresholding in [False, True]:
                    print("=" * 60)
                    print(f"Running {model_name}")
                    print(f"  Cleaning:     {cleaning}")
                    print(f"  Oversampling: {oversampling}")
                    print(f"  Thresholding: {thresholding}")
                    print("=" * 60)

                    train_and_evaluate_model(
                        model_name=model_name,
                        do_cleaning=cleaning,
                        do_oversampling=oversampling,
                        num_epochs=5,
                        batch_size=16,
                        results_file_path=results_file_path,
                        do_thresholding=thresholding,       # <--- use thresholding?
                        class_thresholds=class_thresholds,   # <--- pass thresholds
                        num_labels=6                         # <--- for 6-class problem
                    )


if __name__ == "__main__":
    main()
