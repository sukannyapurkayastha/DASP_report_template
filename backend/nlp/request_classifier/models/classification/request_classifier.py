import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import TrainerCallback, TrainerState, TrainerControl


data_files = {
    "train": "backend/request_classifier/DISAPERE/final_dataset/Request/train.csv",  
    "test": "backend/request_classifier/DISAPERE/final_dataset/Request/test.csv",    
}
data = load_dataset("csv", data_files=data_files)


model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 6 Labels


def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",  # Padding bis zur maximalen Länge
        truncation=True,       # Kürzen von langen Sequenzen
        max_length=128,        # Maximale Sequenzlänge (anpassbar)
        return_attention_mask=True,  # Attention Mask wird erstellt
    )
tokenized_data = data.map(tokenize_function, batched=True)

# 4. Spalte "label" umbenennen und Format auf "torch" setzen
tokenized_data = tokenized_data.rename_column("target", "labels")
tokenized_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = tokenized_data["train"]
test_dataset = tokenized_data["test"]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
    }


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
)


class MetricsLogger(TrainerCallback):
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        print(f"Metrics after epoch {state.epoch}: {metrics}")

# Add the custom callback to the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[MetricsLogger],  # Add the custom callback here
)


trainer.train()


results = trainer.evaluate()
print("Evaluation results:", results)
