import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset, Dataset
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

def clean_special_characters(dataset, column="sentence"):
    """
    Removes special characters such as leading/trailing quotes from the specified column.
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
      - Converts text to lowercase.
    """
    def clean_sentence(example):
        sentence = example['sentence']
        if isinstance(sentence, str):
            sentence = sentence.strip().lower()
            if len(sentence) >= 10 and sentence.count(" ") >= 1:
                return sentence
        return None

    dataset = dataset.filter(lambda example: clean_sentence(example) is not None)
    dataset = dataset.map(lambda example: {"sentence": clean_sentence(example)})
    return dataset

def compute_metrics(eval_pred):
    """
    Computes classification metrics (accuracy, precision, recall, F1).
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
    Logs evaluation metrics after each epoch.
    """
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        print(f"Metrics after epoch {state.epoch}: {metrics}")

def load_data(do_cleaning=True):
    """
    Loads train/validation/test splits from CSV files and optionally cleans them.
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
    Oversamples minority classes to balance class distribution.
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
    Tokenizes text data for BERT.
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
    Loads and optionally cleans/oversamples data, then tokenizes train, validation, and test sets.
    Returns (train_ds, val_ds, test_ds) with "input_ids","attention_mask","labels".
    """
    data = load_data(do_cleaning=do_cleaning)

    if do_oversampling:
        train_dataset = oversample_minority_class(data["train"], label_col="target")
    else:
        train_dataset = data["train"]

    val_dataset = data["validation"]
    test_dataset = data["test"]

    def map_fn(x):
        return tokenize_function(x, tokenizer)

    tokenized_train = train_dataset.map(map_fn, batched=True)
    tokenized_val = val_dataset.map(map_fn, batched=True)
    tokenized_test = test_dataset.map(map_fn, batched=True)

    tokenized_train = tokenized_train.rename_column("target", "labels")
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    tokenized_val = tokenized_val.rename_column("target", "labels")
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    tokenized_test = tokenized_test.rename_column("target", "labels")
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized_train, tokenized_val, tokenized_test

def train_and_evaluate_model(
    model_name,
    do_cleaning=False,
    do_oversampling=False,
    num_epochs=5,
    batch_size=16,
    results_file_path="metrics_results.txt"
):
    """
    Trains a BERT model for binary classification and writes performance metrics to a file.
    """
    if model_name.startswith("model_training/"):
        tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(model_name, local_files_only=True, num_labels=2)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_ds, val_ds, test_ds = prepare_datasets(
        tokenizer,
        do_cleaning=do_cleaning,
        do_oversampling=do_oversampling
    )

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[MetricsLogger()],
    )

    trainer.train()

    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    test_metrics = trainer.evaluate(eval_dataset=test_ds)

    test_output = trainer.predict(test_ds)
    test_preds = np.argmax(test_output.predictions, axis=1)
    cm = confusion_matrix(test_output.label_ids, test_preds)

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

def main():
    """
    Runs multiple training sessions for different settings:
      - Varies model name, data cleaning, and oversampling parameters.
    """
    results_file_path = "metrics_results.txt"
    with open(results_file_path, "w", encoding="utf-8") as f:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Metrics log started at {current_time}\n\n")

    model_names = [
        "allenai/scibert_scivocab_uncased",
        "model_training/nlp/request_classifier/models/classification/sciBERT_neg"
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
