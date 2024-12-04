from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn import BCEWithLogitsLoss
import torch
from sklearn.metrics import f1_score, accuracy_score

# Load the pretrained MLM model
model = BertForSequenceClassification.from_pretrained("../../JitsuPEER_data_and_models_v1/models/bert-base-uncased_neg", num_labels=11, problem_type="multi_label_classification")
# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# Tokenize your dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Assuming dataset has columns 'text' and 'labels' (labels should be a binary vector per example)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Convert dataset to PyTorch tensors
tokenized_datasets = tokenized_datasets.with_format("torch")

# Custom loss function for multi-label classification
class CustomBERTModel(torch.nn.Module):
    def __init__(self, base_model, num_labels):
        super(CustomBERTModel, self).__init__()
        self.bert = base_model
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)  # Pooler output for classification
        loss = None
        if labels is not None:
            loss_fn = BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
        return {"loss": loss, "logits": logits}

# Wrap your fine-tuned model
model = CustomBERTModel(model, num_labels=num_themes)

# Training arguments
training_args = TrainingArguments(
    output_dir="./classification_results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
)

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.sigmoid(torch.tensor(logits)).numpy() > 0.5
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"f1": f1, "accuracy": acc}

# Trainer for classification
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./theme_classification_model")
