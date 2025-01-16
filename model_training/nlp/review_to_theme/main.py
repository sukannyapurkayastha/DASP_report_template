import argparse
import logging
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

# Initialize the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def main(args):
    # Log the received arguments
    logger.info(f"Starting script with arguments: {args}")
    
    # Read input data
    df = pd.read_csv('data_encoded.csv')
    df_train = df[['sentence', 'labels']]
    logger.info("Data loaded successfully.")

    # Convert DataFrame to Dataset
    dataset = Dataset.from_pandas(df_train)
    logger.info("Converted DataFrame to Dataset.")

    # Load the pretrained model
    model = BertForSequenceClassification.from_pretrained(
        args.pretrained_model_path, 
        num_labels=11, 
        problem_type="multi_label_classification"
    )
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    logger.info("Pretrained model and tokenizer loaded successfully.")

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

    dataset = dataset.map(tokenize_function, batched=True)
    logger.info("Dataset tokenization completed.")

    # Train-test split
    dataset = dataset.train_test_split(test_size=0.2, seed=args.seed)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    logger.info("Dataset split into train and test sets.")

    # Define compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(predictions))
        threshold = 0.5
        binary_preds = (probs > threshold).int()
        accuracy = accuracy_score(labels, binary_preds)
        precision = precision_score(labels, binary_preds, average="macro")
        recall = recall_score(labels, binary_preds, average="macro")
        f1 = f1_score(labels, binary_preds, average="macro")
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # Define training arguments
    output_dir = f"./results/pretrained/epoch_{args.epochs}/{args.learning_rate}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.epochs,
        logging_dir="./logs",
        save_strategy="epoch",
        seed=args.seed,
    )

    logger.info("Training arguments defined.")

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed.")

    # Evaluate the model
    eval_results = trainer.evaluate(metric_key_prefix="eval")
    trainer.save_metrics("eval", eval_results)
    logger.info(f"Evaluation results: {eval_results}")

    # Save the model and tokenizer
    final_output_dir = f"./final_model/epoch_{args.epochs}/{args.learning_rate}"
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    logger.info(f"Model and tokenizer saved to {final_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a multi-label classification model using transformers.")

    # Add command-line arguments
    parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training.")

    args = parser.parse_args()

    main(args)
