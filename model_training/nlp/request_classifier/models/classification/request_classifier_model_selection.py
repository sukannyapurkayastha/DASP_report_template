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
        "train": "model_training/nlp/request_classifier/DISAPERE/final_dataset/Request/train.csv",
        "validation": "model_training/nlp/request_classifier/DISAPERE/final_dataset/Request/dev.csv",
        "test": "model_training/nlp/request_classifier/DISAPERE/final_dataset/Request/test.csv",
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
    def map_fn(x): return tokenize_function(x, tokenizer)
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
# 4) Haupt-Training & Vorhersage
#########################

def train_and_evaluate_model(
    model_name,
    do_cleaning=False,
    do_oversampling=False,
    num_epochs=5,
    batch_size=16,
    results_file_path="metrics_results.txt"
):
    """
    Trains a BERT-based model for sequence classification:
    - model_name can be a HF Hub model (e.g. "bert-base-uncased") or a local path (e.g. "classification/custom_model").
    - do_cleaning/do_oversampling toggles for data cleaning and oversampling.

    Writes metrics (Acc, Prec, Recall, F1) and confusion matrix to a txt file.
    """
    # 1) Tokenizer/Model laden
    # Unterscheidung: Lokaler Pfad vs. Hugging-Face-Hub
    if model_name.startswith("model_training/"):
        tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(model_name, local_files_only=True, num_labels=2)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

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

    # 7) Confusion Matrix
    test_output = trainer.predict(test_ds)
    test_preds = np.argmax(test_output.predictions, axis=1)
    cm = confusion_matrix(test_output.label_ids, test_preds)

    # 8) Ergebnisse in Datei schreiben
    with open(results_file_path, "a", encoding="utf-8") as f:
        f.write("==============================================\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Data Cleaning: {do_cleaning}\n")
        f.write(f"Oversampling: {do_oversampling}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write("Validation Metrics:\n")
        f.write(f"  Accuracy:  {val_metrics['eval_accuracy']:.4f}\n")
        f.write(f"  Precision: {val_metrics['eval_precision']:.4f}\n")
        f.write(f"  Recall:    {val_metrics['eval_recall']:.4f}\n")
        f.write(f"  F1-Score:  {val_metrics['eval_f1']:.4f}\n")

        f.write("Test Metrics:\n")
        f.write(f"  Accuracy:  {test_metrics['eval_accuracy']:.4f}\n")
        f.write(f"  Precision: {test_metrics['eval_precision']:.4f}\n")
        f.write(f"  Recall:    {test_metrics['eval_recall']:.4f}\n")
        f.write(f"  F1-Score:  {test_metrics['eval_f1']:.4f}\n")
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
# 5) Hauptprogramm (main)
#########################
def main():
    """
    Führt 16 Trainingsläufe durch:
      - 4 Modelle
      - 2 Einstellungen (do_cleaning: True/False)
      - 2 Einstellungen (do_oversampling: True/False)
    => 4 x 2 x 2 = 16
    """
    results_file_path = "metrics_results.txt"
    # Neue (leere) Datei anlegen oder überschreiben:
    with open(results_file_path, "w", encoding="utf-8") as f:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Ergebnis-Log gestartet am {current_time}\n\n")

    # Unsere 4 Modelle:
    model_names = [                     # 2) BERT-Large
        "allenai/scibert_scivocab_uncased",       # 3) SciBERT
        "model_training/nlp/request_classifier/models/classification/sciBERT_neg"             # 4) Lokales Modell (Beispiel)
    ]

    for model_name in model_names:
        for cleaning in [False, True]:
            for oversampling in [False, True]:
                train_and_evaluate_model(
                    model_name=model_name,
                    do_cleaning=cleaning,
                    do_oversampling=oversampling,
                    num_epochs=5,
                    batch_size=16,
                    results_file_path=results_file_path
                )


if __name__ == "__main__":
    main()
