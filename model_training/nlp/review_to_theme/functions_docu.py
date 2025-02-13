from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Convert logits to predictions for multi-class classification
    probs = torch.sigmoid(torch.tensor(predictions))
    # Convert probabilities to binary predictions using a threshold of 0.5
    threshold = 0.5
    binary_preds = (probs > threshold).int()
    # Calculate metrics
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

def read_epoch_lr_json(base_dirs, df):
    """
    Navigates through subfolders to read JSON files and extract model evaluation metrics.

    This function iterates through a list of base directories, searches for subdirectories 
    representing epoch numbers and learning rates, and reads `all_results.json` files to 
    extract evaluation metrics.

    Args:
        base_dirs (list[str]): List of base directories, each corresponding to a model name.
        df (pd.DataFrame): A DataFrame to which the extracted data will be appended.

    Returns:
        tuple:
            - pd.DataFrame: Updated DataFrame with extracted data.
            - str: Last found epoch value.
            - float: Last found learning rate value.

    The DataFrame will have the following columns:
        - 'model_name': Name of the model (base directory).
        - 'epoch': Epoch number extracted from folder names.
        - 'learning_rate': Learning rate extracted from folder names.
        - 'eval_accuracy': Accuracy metric from JSON file.
        - 'eval_f1': F1-score from JSON file.
        - 'eval_precision': Precision metric from JSON file.
        - 'eval_recall': Recall metric from JSON file.
    """
    epoch_value = None
    lr_value = 0
    for base_dir in base_dirs:
        # Iterate through subfolders A, B, C, D, E, F
        if os.path.exists(base_dir):
            for epoch_folder in os.listdir(base_dir):
                epoch_folder_path = os.path.join(base_dir, epoch_folder)

                # Check if the folder name is an integer (epoch value)
                if os.path.isdir(epoch_folder_path):
                    epoch_value = epoch_folder

                    for lr_folder in os.listdir(epoch_folder_path):
                        lr_folder_path = os.path.join(epoch_folder_path, lr_folder)

                        # Check if the folder name is a valid learning rate (float value)
                        try:
                            lr_value = float(lr_folder)
                        except ValueError:
                            continue  # Skip folders that do not represent learning rate

                        # Now, look for the json file in the learning rate folder
                        json_file_path = os.path.join(lr_folder_path, 'all_results.json')

                        if os.path.isfile(json_file_path):
                            # Read the JSON file
                            with open(json_file_path, 'r') as f:
                                json_data = json.load(f)
                            
                            # Store the result with epoch, learning rate, and json content
                            # Extract necessary values
                            eval_accuracy = json_data.get('eval_accuracy', None)
                            eval_f1 = json_data.get('eval_f1', None)
                            eval_precision = json_data.get('eval_precision', None)
                            eval_recall = json_data.get('eval_recall', None)

                            # Create a new row with the current values
                            new_row = pd.DataFrame([{
                                'model_name': base_dir,  # model_name corresponds to the base_dir
                                'epoch': epoch_value,
                                'learning_rate': lr_value,
                                'eval_accuracy': eval_accuracy,
                                'eval_f1': eval_f1,
                                'eval_precision': eval_precision,
                                'eval_recall': eval_recall
                            }])
                            # Append the new row to the DataFrame
                            df = pd.concat([df, new_row], ignore_index=True)

    return df, epoch_value, lr_value
