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
    Removes special characters (e.g., leading/trailing quotes) from the specified column in the dataset.

    Args:
        dataset (Dataset): A Hugging Face Dataset object.
        column (str, optional): The column name to clean. Defaults to "sentence".

    Returns:
        Dataset: A new dataset with cleaned values in the specified column.
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

    Args:
        dataset (Dataset): A Hugging Face Dataset object.

    Returns:
        Dataset: A new dataset with cleaned sentences.
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




def compute_metrics(eval_pred):
    """
    Computes evaluation metrics for a classification model.

    Args:
        eval_pred (tuple): A tuple containing (logits, labels).

    Returns:
        dict: Dictionary of metric names mapped to their scores.
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




def load_data(do_cleaning=True):
    """
    Loads the dataset from CSV files for train, validation, and test splits,
    and optionally cleans each split.

    Args:
        do_cleaning (bool, optional): Whether to apply data cleaning. Defaults to True.

    Returns:
        DatasetDict: A Hugging Face DatasetDict with train, validation, and test splits.
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
    Oversamples minority classes in the dataset to balance class distribution.

    Args:
        dataset (Dataset): A Hugging Face Dataset.
        label_col (str, optional): The column name containing labels. Defaults to "target".

    Returns:
        Dataset: A new Hugging Face Dataset with balanced class distribution.
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

    Args:
        example (dict): A dictionary with key "sentence" for the text to be tokenized.
        tokenizer (BertTokenizer): A Hugging Face tokenizer instance.

    Returns:
        dict: A dictionary containing input_ids, attention_mask, etc.
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
    Loads, optionally cleans, optionally oversamples, then tokenizes
    the train, validation, and test datasets.

    Args:
        tokenizer (BertTokenizer): A Hugging Face tokenizer.
        do_cleaning (bool, optional): Apply data cleaning if True. Defaults to True.
        do_oversampling (bool, optional): Oversample minority classes if True. Defaults to True.

    Returns:
        tuple: tokenized_train, tokenized_val, tokenized_test (Dataset objects)
    """
    # 1) Load data
    data = load_data(do_cleaning=do_cleaning)

    # 2) Oversampling only on training split
    if do_oversampling:
        train_dataset = oversample_minority_class(data["train"], label_col="target")
    else:
        train_dataset = data["train"]

    val_dataset = data["validation"]
    test_dataset = data["test"]

    # 3) Tokenize
    def map_fn(x): 
        return tokenize_function(x, tokenizer)
    
    tokenized_train = train_dataset.map(map_fn, batched=True)
    tokenized_val = val_dataset.map(map_fn, batched=True)
    tokenized_test = test_dataset.map(map_fn, batched=True)

    # 4) Rename the label column to "labels" and set the format to PyTorch
    tokenized_train = tokenized_train.rename_column("target", "labels")
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    tokenized_val = tokenized_val.rename_column("target", "labels")
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    tokenized_test = tokenized_test.rename_column("target", "labels")
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized_train, tokenized_val, tokenized_test




def apply_thresholds(logits, class_thresholds):
    """
    Applies class-specific thresholds to predictions. For each example:
      1) Convert logits -> softmax probabilities.
      2) For each class, check if p >= class_thresholds[class_idx].
      3) If multiple classes meet their threshold, pick the highest-probability one.
      4) If none meet their threshold, pick the argmax class.

    Args:
        logits (torch.Tensor): Model output logits.
        class_thresholds (dict): A dictionary mapping class_idx -> threshold (e.g. {0: 0.5, 1:0.4, ...}).

    Returns:
        list: A list of predicted class indices.
    """
    probabilities = torch.softmax(torch.tensor(logits), dim=-1)
    predictions = []

    for prob in probabilities:
        valid_classes = []
        for class_idx, p in enumerate(prob):
            # Default threshold is 0.5 if not provided
            if p >= class_thresholds.get(class_idx, 0.5):
                valid_classes.append((class_idx, p))

        if valid_classes:
            predicted_class = max(valid_classes, key=lambda x: x[1])[0]
        else:
            predicted_class = torch.argmax(prob).item()

        predictions.append(predicted_class)
    return predictions


def train_and_evaluate_model(
    model_name,
    do_cleaning=False,
    do_oversampling=False,
    num_epochs=5,
    batch_size=16,
    results_file_path="metrics_results.txt",
    do_thresholding=False,
    class_thresholds=None,
    num_labels=6
):
    """
    Trains a BERT-based model for sequence classification.

    Args:
        model_name (str): Model identifier, either on Hugging Face Hub (e.g., "bert-base-uncased")
                          or a local path (e.g., "classification/custom_model").
        do_cleaning (bool, optional): Whether to clean the data. Defaults to False.
        do_oversampling (bool, optional): Whether to oversample minority classes. Defaults to False.
        num_epochs (int, optional): Number of training epochs. Defaults to 5.
        batch_size (int, optional): Training/evaluation batch size. Defaults to 16.
        results_file_path (str, optional): Path to write evaluation metrics. Defaults to "metrics_results.txt".
        do_thresholding (bool, optional): Whether to apply class-specific thresholds. Defaults to False.
        class_thresholds (dict, optional): Threshold values per class index. Defaults to None.
        num_labels (int, optional): Number of output classes. Defaults to 6.
    """
    # 1) Load tokenizer and model (distinguish local path vs. HF Hub)
    if model_name.startswith("model_training/"):
        tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(model_name, local_files_only=True, num_labels=num_labels)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # 2) Prepare datasets
    train_ds, val_ds, test_ds = prepare_datasets(
        tokenizer,
        do_cleaning=do_cleaning,
        do_oversampling=do_oversampling
    )

    # 3) Training Arguments
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

    # 4) Instantiate the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[MetricsLogger()],
    )

    # 5) Train the model
    trainer.train()

    # 6) Evaluate on validation and test sets
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    test_metrics = trainer.evaluate(eval_dataset=test_ds)

    # 7) Generate Predictions and Confusion Matrix
    test_output = trainer.predict(test_ds)
    pred_logits = test_output.predictions
    true_labels = test_output.label_ids

    if do_thresholding and class_thresholds is not None:
        # Use custom thresholding
        pred_labels = apply_thresholds(pred_logits, class_thresholds)
        cm = confusion_matrix(true_labels, pred_labels)
    else:
        pred_labels = np.argmax(pred_logits, axis=-1)
        cm = confusion_matrix(true_labels, pred_labels)

    # 8) Write results to file
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

    Args:
        texts (list): A list of text inputs to classify.
        model_path (str, optional): Path or name of the model checkpoint. Defaults to "./bert_request_classifier_model".

    Returns:
        np.ndarray: An array of predicted label indices.
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
    Main function to orchestrate training over multiple configurations.
    Writes results (metrics and confusion matrices) to a specified file.
    """
    results_file_path = "metrics_results.txt"
    
    # Start fresh results file
    with open(results_file_path, "w", encoding="utf-8") as f:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Metrics log started at {current_time}\n\n")

    # List of model names or paths to evaluate
    model_names = [
        "model_training/nlp/request_classifier/models/classification/sciBERT_neg/"
    ]

    # Example thresholds for 6 classes
    class_thresholds = {
        0: 0.35,
        1: 0.35,
        2: 0.25,
        3: 0.20,
        4: 0.25,
        5: 0.35
    }

    # Define different parameter combinations to test
    missing_combos = [
        (True, False, True),   # cleaning=True, oversampling=False, thresholding=True
        (True, True, False),   # cleaning=True, oversampling=True,  thresholding=False
        (True, True, True)     # cleaning=True, oversampling=True,  thresholding=True
    ]

    for model_name in model_names:
        for (cleaning, oversampling, thresholding) in missing_combos:
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
                do_thresholding=thresholding,
                class_thresholds=class_thresholds,
                num_labels=6
            )


if __name__ == "__main__":
    main()
