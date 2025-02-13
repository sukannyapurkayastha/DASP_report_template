from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import torch  # Ensure torch is imported for tensor operations

def plot_confusion_matrix(model, tokenizer, test_df, title="Confusion Matrix"):
    """
    Generates and displays a confusion matrix for the model's predictions on the test dataset.

    Parameters:
        model: The trained model used for predictions.
        tokenizer: The tokenizer corresponding to the model.
        test_df (DataFrame): The test dataset containing 'data' (text) and 'encoded_cat' (integer labels).
        title (str): Title for the confusion matrix plot. Default is "Confusion Matrix".
    """
    preds = []  # List to store predicted labels
    test_texts = test_df["data"].tolist()  # Extract test texts
    test_labels_list = test_df["encoded_cat"].tolist()  # Extract true labels

    # Iterate over test texts and generate predictions
    for txt in test_texts:
        # Tokenize the text input and convert to tensor format
        inputs = tokenizer(txt, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move inputs to the model's device (CPU/GPU)

        # Perform inference without gradient computation
        with torch.no_grad():
            outputs = model(**inputs)  # Get model predictions

        logits = outputs.logits  # Extract logits from model output
        pred_label = torch.argmax(logits, dim=1).item()  # Get the predicted class label
        preds.append(pred_label)

    # Compute the confusion matrix
    cm = confusion_matrix(test_labels_list, preds)

    # Define class names for the plot
    class_names = [
        "none", "substance", "originality", "clarity", 
        "soundness-correctness", "motivation-impact", 
        "meaningful-comparison", "replicability", "other"
    ]

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(10, 8))  # Set figure size
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", 
        xticklabels=class_names, yticklabels=class_names
    )  # Display the matrix with annotations
    plt.title(title)  # Set the title
    plt.ylabel("True Label")  # Label for y-axis
    plt.xlabel("Predicted Label")  # Label for x-axis
    plt.show()  # Display the plot

    # Print the classification report
    print(classification_report(test_labels_list, preds, target_names=class_names))
def train_approach_1(model_class, tokenizer_class, model_path, model_name):
    """
    Ansatz (1): Train for 2 epochs without oversampling.
    
    Parameters:
        model_class: The class for the Hugging Face model.
        tokenizer_class: The class for the Hugging Face tokenizer.
        model_path: Path to the pretrained model.
        model_name: Name of the model (used for saving results).
    
    Returns:
        model: The trained model.
        tokenizer: The tokenizer used for training.
        trainer: The Hugging Face Trainer instance.
    """
    out_dir = f"./results_approach1_{model_name}"  # Dynamic folder for saving results

    # Load the pretrained model and tokenizer
    model = model_class.from_pretrained(
        model_path,
        num_labels=len(all_labels_list),                 # Number of classes for classification
        problem_type="single_label_classification",      # Single-label classification problem
        ignore_mismatched_sizes=True                    # Allow head size mismatch
    )
    tokenizer = tokenizer_class.from_pretrained(model_path)

    # Create a Trainer with specified output directory
    trainer = create_trainer(model, tokenizer, train_df, val_df, num_epochs=4, output_dir=out_dir)
    trainer.train()  # Train the model

    # Evaluate on the test set
    test_encodings, test_labels = encode_data(tokenizer, test_df)
    test_dataset = CustomDataset(test_encodings, test_labels)
    final_eval_results = trainer.evaluate(test_dataset)
    print("Ansatz (1) - Final Test Results:", final_eval_results)

    # Save the final model and tokenizer
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    return model, tokenizer, trainer


def train_approach_2(model_class, tokenizer_class, model_path, model_name):
    """
    Ansatz (2): Train for 2 epochs with oversampling.
    
    Parameters:
        model_class: The class for the Hugging Face model.
        tokenizer_class: The class for the Hugging Face tokenizer.
        model_path: Path to the pretrained model.
        model_name: Name of the model (used for saving results).
    
    Returns:
        model: The trained model.
        tokenizer: The tokenizer used for training.
        trainer: The Hugging Face Trainer instance.
    """
    # Oversample the training dataset
    oversampled_train_df = oversample_dataframe(train_df)
    out_dir = f"./results_approach2_{model_name}"

    # Load the pretrained model and tokenizer
    model = model_class.from_pretrained(
        model_path,
        num_labels=len(all_labels_list),
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True
    )
    tokenizer = tokenizer_class.from_pretrained(model_path)

    # Create a Trainer with the oversampled dataset
    trainer = create_trainer(model, tokenizer, oversampled_train_df, val_df, num_epochs=4, output_dir=out_dir)
    trainer.train()  # Train the model

    # Evaluate on the test set
    test_encodings, test_labels = encode_data(tokenizer, test_df)
    test_dataset = CustomDataset(test_encodings, test_labels)
    final_eval_results = trainer.evaluate(test_dataset)
    print("Ansatz (2) - Final Test Results:", final_eval_results)

    # Save the final model and tokenizer
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    return model, tokenizer, trainer


def train_approach_3(model_class, tokenizer_class, model_path, model_name):
    """
    Ansatz (3): Train for 1 epoch with oversampling followed by 1 epoch without oversampling.
    
    Parameters:
        model_class: The class for the Hugging Face model.
        tokenizer_class: The class for the Hugging Face tokenizer.
        model_path: Path to the pretrained model.
        model_name: Name of the model (used for saving results).
    
    Returns:
        model: The trained model after both phases.
        tokenizer: The tokenizer used for training.
        trainer: The Hugging Face Trainer instance after phase B.
    """
    out_dir_stepA = f"./results_approach3_stepA_{model_name}"  # Folder for Step A results
    out_dir_stepB = f"./results_approach3_stepB_{model_name}"  # Folder for Step B results

    # Step A: Train with oversampled dataset
    oversampled_train_df = oversample_dataframe(train_df)
    model_stepA = model_class.from_pretrained(
        model_path,
        num_labels=len(all_labels_list),
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True
    )
    tokenizer_stepA = tokenizer_class.from_pretrained(model_path)

    trainer_stepA = create_trainer(model_stepA, tokenizer_stepA, oversampled_train_df, val_df, num_epochs=2, output_dir=out_dir_stepA)
    trainer_stepA.train()  # Train the model for Step A

    # Step B: Continue training with the original (non-oversampled) dataset
    trainer_stepB = create_trainer(model_stepA, tokenizer_stepA, train_df, val_df, num_epochs=2, output_dir=out_dir_stepB)
    trainer_stepB.train()  # Train the model for Step B

    # Evaluate on the test set
    test_encodings, test_labels = encode_data(tokenizer_stepA, test_df)
    test_dataset = CustomDataset(test_encodings, test_labels)
    final_eval_results = trainer_stepB.evaluate(test_dataset)
    print("Ansatz (3) - Final Test Results:", final_eval_results)

    # Save the final model and tokenizer after Step B
    trainer_stepB.save_model(out_dir_stepB)
    tokenizer_stepA.save_pretrained(out_dir_stepB)

    return model_stepA, tokenizer_stepA, trainer_stepB



# =========================================
# 2) Hilfsfunktionen
# =========================================

def oversample_dataframe(input_df, skip_label="None"):
    """
    Einfache Oversampling-Funktion, die jede Klasse (außer skip_label) um den Faktor 2 erhöht.
    Passe dies nach Bedarf an (z.B. max_count, SMOTE etc.).
    
    Parameters:
        input_df (DataFrame): Eingabedaten mit 'labels'-Spalte.
        skip_label (str): Label, das vom Oversampling ausgeschlossen wird. Standard ist "None".
    
    Returns:
        DataFrame: Oversampled DataFrame mit verdoppelten Einträgen für alle relevanten Klassen.
    """
    aspects = [lbl for lbl in all_labels_list if lbl != skip_label]  # Exclude specific label from oversampling
    output_df = input_df.copy()

    for aspect in aspects:
        subset = output_df[output_df["labels"] == aspect]  # Filter rows for the specific label
        count = len(subset)  # Count occurrences of the label
        if count > 0:
            # Duplicate samples for oversampling
            resampled = subset.sample(n=count, replace=True, random_state=42)
            output_df = pd.concat([output_df, resampled], ignore_index=True)
    return output_df


def encode_data(tokenizer, df):
    """
    Tokenisiert die Texte und gibt Encodings sowie Labels als Torch Tensor zurück.
    
    Parameters:
        tokenizer: Tokenizer-Instanz für die Textverarbeitung.
        df (DataFrame): Daten mit 'data' (Text) und 'encoded_cat' (Integer-Labels).
    
    Returns:
        Tuple: (Encodings, Labels-Tensor).
    """
    texts = df["data"].tolist()  # Extract text data
    labels = df["encoded_cat"].tolist()  # Extract encoded labels

    # Tokenize the text data with truncation and padding
    encodings = tokenizer(texts, truncation=True, padding=True)
    labels_tensor = torch.tensor(labels, dtype=torch.long)  # Convert labels to Torch tensor

    return encodings, labels_tensor


class CustomDataset(Dataset):
    """
    Einfaches Dataset für den Trainer.
    
    Parameters:
        encodings (dict): Tokenisierte Textdaten.
        labels (Tensor): Labels in Tensor-Form.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Gibt ein einzelnes Item (Encoding + Label) zurück.
        
        Parameters:
            idx (int): Index des gewünschten Eintrags.
        
        Returns:
            dict: Tokenisierte Eingaben und zugehöriges Label.
        """
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}  # Convert encodings to tensor
        item["labels"] = self.labels[idx]  # Add label to the dictionary
        return item

    def __len__(self):
        """
        Gibt die Gesamtzahl der Einträge im Dataset zurück.
        
        Returns:
            int: Anzahl der Datenpunkte.
        """
        return len(self.labels)


def compute_metrics(pred):
    """
    Berechnet die Accuracy als Beispielmetrik.
    
    Parameters:
        pred: Predictions-Objekt des Modells.
    
    Returns:
        dict: Dictionary mit dem Schlüssel 'accuracy' und dem berechneten Wert.
    """
    labels = pred.label_ids  # True labels
    preds = np.argmax(pred.predictions, axis=1)  # Predicted labels
    accuracy = (preds == labels).mean()  # Calculate accuracy
    return {"accuracy": accuracy}


def create_trainer(model, tokenizer, train_df, val_df, num_epochs=4, output_dir="./results"):
    """
    Erstellt einen Hugging Face Trainer mit spezifiziertem Modell, Tokenizer und Daten.
    
    Parameters:
        model: Pretrained Modell-Instanz.
        tokenizer: Tokenizer-Instanz für Textverarbeitung.
        train_df (DataFrame): Trainingsdaten.
        val_df (DataFrame): Validierungsdaten.
        num_epochs (int): Anzahl der Trainings-Epochen. Standard ist 4.
        output_dir (str): Speicherort für Trainingsergebnisse. Standard ist "./results".
    
    Returns:
        Trainer: Hugging Face Trainer-Objekt.
    """
    # Tokenize and encode training and validation datasets
    train_encodings, train_labels = encode_data(tokenizer, train_df)
    val_encodings, val_labels = encode_data(tokenizer, val_df)

    # Create PyTorch Datasets
    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,                 # Output directory for results
        evaluation_strategy="epoch",          # Evaluate after each epoch
        save_strategy="epoch",                # Save model after each epoch
        learning_rate=5e-5,                   # Learning rate
        per_device_train_batch_size=16,       # Batch size for training
        per_device_eval_batch_size=64,        # Batch size for evaluation
        num_train_epochs=num_epochs,          # Number of epochs
        weight_decay=0.01,                    # Weight decay for regularization
        logging_dir='./logs',                 # Directory for logs
        load_best_model_at_end=True,          # Load the best model at the end of training
        metric_for_best_model="accuracy",     # Metric to determine the best model
        greater_is_better=True,               # Higher metric values indicate better performance
        save_total_limit=2                    # Keep only the last 2 model checkpoints
    )

    # Define early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,            # Stop training after 3 epochs with no improvement
        early_stopping_threshold=0.0         # Minimum improvement threshold
    )

    # Create the Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,      # Use custom metrics
        callbacks=[early_stopping]            # Add early stopping callback
    )

    return trainer
