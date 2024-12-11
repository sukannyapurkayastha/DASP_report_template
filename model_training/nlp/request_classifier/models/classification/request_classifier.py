import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import TrainerCallback, TrainerState, TrainerControl
import pandas as pd
import os
import numpy as np

def clean_special_characters(dataset, column="sentence"):
    """
    Removes special characters like leading/trailing quotes from the specified column in the dataset.

    Parameters:
    dataset (Dataset): A Hugging Face Dataset object.
    column (str): The column to clean (default: "sentence").

    Returns:
    Dataset: Cleaned dataset with special characters removed from the column.
    """
    def clean_example(example):
        example[column] = example[column].strip().replace('"', '') if isinstance(example[column], str) else example[column]
        return example

    # Apply the cleaning function to each example in the dataset
    return dataset.map(clean_example)

def clean_dataset(dataset):
    """
    Cleans and filters a Hugging Face Dataset:
    - Removes sentences shorter than 10 characters or with fewer than 1 whitespace.
    - Converts all text to lowercase.

    Parameters:
    dataset (Dataset): A Hugging Face Dataset object.

    Returns:
    Dataset: Cleaned dataset with filtered and processed sentences.
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

    Parameters:
    eval_pred (tuple): A tuple containing logits and labels.

    Returns:
    dict: A dictionary with accuracy, F1 score, precision, and recall.
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

def load_and_clean_data():
    """
    Loads and cleans the dataset from CSV files.

    Returns:
    DatasetDict: A dictionary containing train, validation, and test datasets.
    """
    data_files = {
        "train": "model_training/nlp/request_classifier/DISAPERE/final_dataset/Request/train.csv",
        "validation": "model_training/nlp/request_classifier/DISAPERE/final_dataset/Request/dev.csv",
        "test": "model_training/nlp/request_classifier/DISAPERE/final_dataset/Request/test.csv",
    }

    # Load datasets
    data = load_dataset("csv", data_files=data_files)

    # Clean datasets
    for split in ["train", "validation", "test"]:
        data[split] = clean_dataset(data[split])
        data[split] = clean_special_characters(data[split], "sentence")

    return data

def oversample_minority_class(dataset, label_col="target"):
    """
    Oversamples the minority classes in the dataset to balance the class distribution.

    Parameters:
    dataset (Dataset): A Hugging Face Dataset object.
    label_col (str): The column containing class labels (default: "target").

    Returns:
    Dataset: A balanced dataset with oversampled minority classes.
    """
    df = dataset.to_pandas()
    class_counts = df[label_col].value_counts()
    max_count = class_counts.max()

    # Oversample each class to match the majority class size
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

    Parameters:
    example (dict): A dictionary containing text data.
    tokenizer (BertTokenizer): A tokenizer object.

    Returns:
    dict: Tokenized data.
    """
    return tokenizer(
        example["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_attention_mask=True,
    )

def prepare_datasets(tokenizer):
    """
    Prepares and tokenizes datasets for training, validation, and testing.

    Parameters:
    tokenizer (BertTokenizer): A tokenizer object.

    Returns:
    tuple: Tokenized train, validation, and test datasets.
    """
    data = load_and_clean_data()

    # Oversample the training dataset
    train_dataset = oversample_minority_class(data["train"], label_col="target")
    validation_dataset = data["validation"]
    test_dataset = data["test"]

    # Tokenize datasets
    tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_test = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Rename label column for compatibility
    for dataset in [tokenized_train, tokenized_validation, tokenized_test]:
        dataset = dataset.rename_column("target", "labels")
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized_train, tokenized_validation, tokenized_test

def train_model():
    """
    Trains a BERT-based model for sequence classification.

    - Loads and tokenizes datasets.
    - Configures training arguments and starts training.
    - Evaluates the model on validation and test datasets.
    - Saves the trained model and tokenizer.

    Returns:
    None
    """
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_dataset, validation_dataset, test_dataset = prepare_datasets(tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        callbacks=[MetricsLogger()],
    )

    # Train the model
    trainer.train()

    # Evaluate on validation and test datasets
    print("Validation results:", trainer.evaluate(eval_dataset=validation_dataset))
    print("Test results:", trainer.evaluate(eval_dataset=test_dataset))

    # Generate confusion matrix
    test_output = trainer.predict(test_dataset)
    test_preds = np.argmax(test_output.predictions, axis=1)
    cm = confusion_matrix(test_output.label_ids, test_preds)
    print("Confusion Matrix:\n", cm)

    # Save the model and tokenizer
    save_directory = os.path.join(os.path.dirname(__file__), "../../../../../backend/models/request_classifier/request_classifier")
    trainer.save_model(save_directory)
    tokenizer.save_pretrained(save_directory)

def predict(texts, model_path="./bert_request_classifier_model"):
    """
    Predicts labels for a list of texts using a trained BERT model.

    Parameters:
    texts (list of str): List of input texts to classify.
    model_path (str): Path to the trained model.

    Returns:
    numpy.ndarray: Predicted labels for the input texts.
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

if __name__ == "__main__":
    train_model()
