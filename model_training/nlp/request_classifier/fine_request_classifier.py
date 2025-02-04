import torch
from transformers import DebertaTokenizer, DebertaForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load the dataset from CSV files
data_files = {
    "train": "backend/nlp/request_classifier/DISAPERE/final_dataset/fine_request/train.csv",
    "validation": "backend/nlp/request_classifier/DISAPERE/final_dataset/fine_request/test.csv",
    "test": "backend/nlp/request_classifier/DISAPERE/final_dataset/fine_request/dev.csv"
}
data = load_dataset("csv", data_files=data_files)

model_name = "microsoft/deberta-xlarge"
tokenizer = DebertaTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    """
    Tokenizes input text for the DeBERTa model.

    Args:
        example (dict): A dictionary containing text data under the key "text".

    Returns:
        dict: Tokenized output with padding, truncation, and attention masks.
    """
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_attention_mask=True,
    )

# Convert training split into a DataFrame for oversampling
df_train = pd.DataFrame({
    "text": [example["text"] for example in data["train"]],
    "label": [example["target"] for example in data["train"]]
})

# Oversample the minority classes in the training set
oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(df_train[["text"]], df_train["label"])

# Create a Hugging Face Dataset from the resampled data
resampled_train_dataset = Dataset.from_pandas(
    pd.DataFrame({"text": X_resampled["text"], "target": y_resampled})
)

# Convert validation and test splits into Dataset objects
validation_dataset = Dataset.from_pandas(pd.DataFrame({
    "text": [example["text"] for example in data["validation"]],
    "target": [example["target"] for example in data["validation"]]
}))
test_dataset = Dataset.from_pandas(pd.DataFrame({
    "text": [example["text"] for example in data["test"]],
    "target": [example["target"] for example in data["test"]]
}))

# Combine all splits into a DatasetDict
resampled_data = DatasetDict({
    "train": resampled_train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})

# Tokenize all splits
tokenized_resampled_data = resampled_data.map(tokenize_function, batched=True)
tokenized_resampled_data = tokenized_resampled_data.rename_column("target", "labels")
tokenized_resampled_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Separate into individual datasets
train_dataset = tokenized_resampled_data["train"]
val_dataset = tokenized_resampled_data["validation"]
test_dataset = tokenized_resampled_data["test"]

# Compute class weights for handling class imbalance
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_resampled),
    y=y_resampled
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class Weights:", class_weights)

# Define a penalty matrix for specific misclassifications
penalty_matrix = torch.tensor([
    [1.0, 1.0, 1.0, 1.0, 1.5, 1.5],  # Class 0
    [1.0, 1.0, 1.5, 1.0, 1.5, 1.5],  # Class 1
    [1.0, 1.5, 1.0, 1.5, 1.0, 1.5],  # Class 2
    [1.0, 1.0, 1.5, 1.0, 1.0, 1.5],  # Class 3
    [1.5, 1.0, 1.0, 1.0, 1.0, 1.5],  # Class 4
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Class 5
], dtype=torch.float)

class CustomLossWithPenalties(torch.nn.Module):
    """
    Custom loss function that combines cross-entropy with a penalty matrix.

    Args:
        penalty_matrix (torch.Tensor): A matrix specifying misclassification penalties.
        class_weights (torch.Tensor): Class weights for handling imbalance.
    """
    def __init__(self, penalty_matrix, class_weights):
        super().__init__()
        self.penalty_matrix = penalty_matrix
        self.class_weights = class_weights

    def forward(self, logits, labels):
        ce_loss = torch.nn.functional.cross_entropy(logits, labels, weight=self.class_weights, reduction="none")
        penalties = self.penalty_matrix[labels, torch.argmax(logits, dim=-1)]
        loss = ce_loss * penalties
        return loss.mean()

class PenalizedDebertaForSequenceClassification(DebertaForSequenceClassification):
    """
    Custom DeBERTa model that incorporates a penalty matrix into the loss calculation.

    Args:
        penalty_matrix (torch.Tensor): Matrix specifying misclassification penalties.
        class_weights (torch.Tensor): Class weights for handling imbalance.
    """
    def __init__(self, config, penalty_matrix, class_weights):
        super().__init__(config)
        self.penalty_matrix = penalty_matrix
        self.class_weights = class_weights

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        logits = outputs.logits

        if labels is not None:
            loss_fct = CustomLossWithPenalties(self.penalty_matrix, self.class_weights)
            loss = loss_fct(logits, labels)
            outputs = (loss, logits) + outputs[2:]

        return outputs

# Initialize the custom model
model = PenalizedDebertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=6,
    penalty_matrix=penalty_matrix,
    class_weights=class_weights
)

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

def compute_metrics(eval_pred):
    """
    Computes evaluation metrics for the model.

    Args:
        eval_pred (tuple): A tuple containing (logits, labels).

    Returns:
        dict: A dictionary with accuracy, f1, precision, and recall.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the custom model
trainer.train()

# Evaluate on validation set
val_results = trainer.evaluate(eval_dataset=val_dataset)
print("Validation results:", val_results)

# Evaluate on test set
test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test results:", test_results)

# Predictions with class-specific thresholds
predictions = trainer.predict(test_dataset)
pred_logits = predictions.predictions
true_labels = predictions.label_ids

class_thresholds = {
    0: 0.35,
    1: 0.35,
    2: 0.25,
    3: 0.2,
    4: 0.25,
    5: 0.35
}

def apply_thresholds(logits, class_thresholds):
    """
    Applies class-specific thresholds to predictions.

    Args:
        logits (numpy.ndarray): The model's output logits.
        class_thresholds (dict): Threshold values for each class index.

    Returns:
        list: A list of predicted class labels after thresholding.
    """
    probabilities = torch.softmax(torch.tensor(logits), dim=-1)
    pred_labels = []

    for prob in probabilities:
        valid_classes = []
        for class_idx, p in enumerate(prob):
            if p >= class_thresholds.get(class_idx, 0.5):
                valid_classes.append((class_idx, p))

        if valid_classes:
            predicted_class = max(valid_classes, key=lambda x: x[1])[0]
        else:
            predicted_class = torch.argmax(prob).item()
        pred_labels.append(predicted_class)

    return pred_labels

pred_labels = apply_thresholds(pred_logits, class_thresholds)

# Confusion Matrix visualization
conf_matrix = confusion_matrix(true_labels, pred_labels, labels=list(class_thresholds.keys()))
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=list(class_thresholds.keys()),
    yticklabels=list(class_thresholds.keys())
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
