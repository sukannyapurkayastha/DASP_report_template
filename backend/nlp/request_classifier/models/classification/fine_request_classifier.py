import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import TrainerCallback, TrainerState, TrainerControl
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datasets import Dataset
from collections import Counter


# Daten laden
data_files = {"train" : "backend/request_classifier/DISAPERE/final_dataset/fine_request/train.csv", 
             "test" : "backend/request_classifier/DISAPERE/final_dataset/fine_request/test.csv"}
data = load_dataset("csv", data_files=data_files)

# Modell und Tokenizer initialisieren
model_name = "roberta-base"  # Alternativen: "roberta-large", "cardiffnlp/twitter-roberta-base-sentiment"

# Tokenizer und Modell laden
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=6)   # Labels

# Tokenizer Funktion
def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",  # Padding bis zur maximalen Länge
        truncation=True,       # Kürzen von langen Sequenzen
        max_length=128,        # Maximale Sequenzlänge (anpassbar)
        return_attention_mask=True,  # Attention Mask wird erstellt
    )

# Tokenize die Daten
tokenized_data = data.map(tokenize_function, batched=True)

# Sampling vorbereiten
df_train = pd.DataFrame({
    "text": [example["text"] for example in data["train"]],
    "label": [example["target"] for example in data["train"]]
})

original_distribution = Counter(df_train["label"])
print("Original-Klassenverteilung:", original_distribution)

# Gesamtanzahl der Texte
total_texts = len(df_train)

# Zielverteilung definieren basierend auf der Gesamtanzahl,
# aber sicherstellen, dass jede Klasse mindestens so viele Samples wie ursprünglich hat
target_distribution = {
    label: max(original_count, int(total_texts / (5 + i)))  # Sicherstellen, dass kein Wert kleiner ist als original
    for i, (label, original_count) in enumerate(original_distribution.items())
}

print("Zielverteilung:", target_distribution)

# Oversampler mit angepasster Strategie
oversampler = RandomOverSampler(sampling_strategy=target_distribution)
X_resampled, y_resampled = oversampler.fit_resample(df_train[["text"]], df_train["label"])

# Neue Klassenverteilung anzeigen
new_distribution = Counter(y_resampled)
print("Neue-Klassenverteilung nach Resampling:", new_distribution)

# Resampelte Daten als Dataset
resampled_train_dataset = Dataset.from_pandas(pd.DataFrame({"text": X_resampled["text"], "target": y_resampled}))

# Resampelte Daten als `DatasetDict`
resampled_data = DatasetDict({
    "train": resampled_train_dataset,
    "test": data["test"]
})

# Tokenize die resampelten Daten
tokenized_resampled_data = resampled_data.map(tokenize_function, batched=True)
tokenized_resampled_data = tokenized_resampled_data.rename_column("target", "labels")
tokenized_resampled_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = tokenized_resampled_data["train"]
test_dataset = tokenized_resampled_data["test"]

# Metriken definieren
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
    }

# Training Argumente
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

# Callback für Logging
class MetricsLogger(TrainerCallback):
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        print(f"Metrics after epoch {state.epoch}: {metrics}")

# Trainer initialisieren
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[MetricsLogger],  # Callback hinzufügen
)

# Training starten
trainer.train()

# Evaluieren
results = trainer.evaluate()
print("Evaluation results:", results)

# Vorhersagen machen
predictions = trainer.predict(test_dataset)
pred_logits = torch.tensor(predictions.predictions)
pred_labels = torch.argmax(pred_logits, dim=1).numpy()
true_labels = predictions.label_ids

# Confusion Matrix erstellen
conf_matrix = confusion_matrix(true_labels, pred_labels)

# Confusion Matrix visualisieren
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2, 3, 4, 5], yticklabels=[0, 1, 2, 3, 4, 5])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
