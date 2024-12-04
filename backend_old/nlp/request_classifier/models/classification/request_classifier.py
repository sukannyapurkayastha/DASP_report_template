import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import TrainerCallback, TrainerState, TrainerControl

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
        "train": "backend/nlp/request_classifier/DISAPERE/final_dataset/Request/train.csv",
        "validation": "backend/nlp/request_classifier/DISAPERE/final_dataset/Request/dev.csv",
        "test": "backend/nlp/request_classifier/DISAPERE/final_dataset/Request/test.csv",
    }
    data = load_dataset("csv", data_files=data_files)
    return data


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
    tokenized_data = data.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_data = tokenized_data.rename_column("target", "labels")
    tokenized_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    train_dataset = tokenized_data["train"]
    validation_dataset = tokenized_data["validation"]
    test_dataset = tokenized_data["test"]
    return train_dataset, validation_dataset, test_dataset


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
        num_train_epochs=1,
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
        callbacks=[MetricsLogger],
    )

    trainer.train()

    validation_results = trainer.evaluate(eval_dataset=validation_dataset)
    print("Validation results:", validation_results)

    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print("Test results:", test_results)

    # Save the model
    trainer.save_model("./bert_request_classifier_model")


def predict(texts, model_path="./bert_request_classifier_model"):
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
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
