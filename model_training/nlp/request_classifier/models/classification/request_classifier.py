import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import TrainerCallback, TrainerState, TrainerControl
import pandas as pd
import os

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
    }

class MetricsLogger(TrainerCallback):
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        print(f"Metrics after epoch {state.epoch}: {metrics}")

def load_data():
    data_files = {
        "train": "model_training/nlp/request_classifier/DISAPERE/final_dataset/Request/train.csv",
        "validation": "model_training/nlp/request_classifier/DISAPERE/final_dataset/Request/dev.csv",
        "test": "model_training/nlp/request_classifier/DISAPERE/final_dataset/Request/test.csv",
    }
    data = load_dataset("csv", data_files=data_files)
    return data

def undersample_majority_class(dataset, label_col="target"):
    # Convert dataset to a pandas DataFrame
    df = dataset.to_pandas()
    # Get class counts
    class_counts = df[label_col].value_counts()
    # Identify the smallest class and its count
    minority_class = class_counts.idxmin()
    minority_count = class_counts.min()

    # Resample each class to the minority_count
    balanced_dfs = []
    for cls in class_counts.index:
        df_cls = df[df[label_col] == cls]
        if cls == minority_class:
            balanced_dfs.append(df_cls)
        else:
            # Undersample
            balanced_dfs.append(df_cls.sample(minority_count, random_state=42))

    # Concatenate and shuffle
    balanced_df = pd.concat(balanced_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Convert back to a Dataset
    balanced_dataset = Dataset.from_pandas(balanced_df, preserve_index=False)
    return balanced_dataset

def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_attention_mask=True,
    )

def prepare_datasets(tokenizer):
    data = load_data()
    # Undersample the training set
    train_dataset = undersample_majority_class(data["train"], label_col="target")
    # Keep validation and test as is
    validation_dataset = data["validation"]
    test_dataset = data["test"]

    tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_test = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    tokenized_train = tokenized_train.rename_column("target", "labels")
    tokenized_validation = tokenized_validation.rename_column("target", "labels")
    tokenized_test = tokenized_test.rename_column("target", "labels")

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_validation.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized_train, tokenized_validation, tokenized_test

def train_model():
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
        num_train_epochs=4,
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

    trainer.train()

    validation_results = trainer.evaluate(eval_dataset=validation_dataset)
    print("Validation results:", validation_results)

    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print("Test results:", test_results)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    save_directory = os.path.join(script_directory, "../../../../../backend/models/request_classifier/request_classifier")

    # Save the model
    trainer.save_model(save_directory)
    # Save the tokenizer
    tokenizer.save_pretrained(save_directory)

def predict(texts, model_path="./bert_request_classifier_model"):
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
    predictions = torch.argmax(logits, dim=-1)
    return predictions.cpu().numpy()

if __name__ == "__main__":
    train_model()
