import os
import json
import pandas as pd

# Define the base directories
#"/mnt/beegfs/work/yang1/attitude_theme_classifier/results/scibert_all/scibert_scivocab_uncased/"
base_dirs = ['bert-base-uncased_all', 'bert-base-uncased_neg', 'roberta-base_all', 'roberta-base_neg',
             'scibert_all/scibert_scivocab_uncased', 'scibert_neg/scibert_scivocab_uncased']

# Function to navigate through subfolders and read the JSON files
def read_epoch_lr_json(base_dirs, df):
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

# Run the function and print the results
# Initialize an empty DataFrame with the specified columns
columns = ['model_name', 'epoch', 'learning_rate', 'eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall']
df = pd.DataFrame(columns=columns)
results, e, lr = read_epoch_lr_json(base_dirs,df)
results.to_csv(f'{e}_{lr}_eval_results.csv', index=False)
