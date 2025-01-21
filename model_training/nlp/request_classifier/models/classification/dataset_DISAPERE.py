import os
import json
import pandas as pd



def merge_json_files(folder_path, output_file):
    """
    Merges all JSON files within the specified folder into a single JSON file.

    Args:
        folder_path (str): The directory path containing JSON files to merge.
        output_file (str): The output JSON file path.

    Returns:
        None
    """
    merged_data = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                merged_data.append(data)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    try:
        with open(output_file, 'w', encoding='utf-8') as output_json_file:
            json.dump(merged_data, output_json_file, ensure_ascii=False, indent=4)
        print(f"Merged JSON file created: {output_file}")
    except Exception as e:
        print(f"Error writing {output_file}: {e}")



def add_unique_target_index_to_nested_list(input_file, output_file):
    """
    Adds a unique 'target_index' field to each sentence in 'review_sentences',
    based on the 'review_action'. Each unique action is mapped to a unique index.

    Args:
        input_file (str): The input JSON file containing objects with 'review_sentences'.
        output_file (str): The output JSON file to write the modified data.

    Returns:
        None
    """
    with open(input_file, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    action_to_index = {}
    current_index = 0

    for item in data:
        for sentence in item.get("review_sentences", []):
            action = sentence.get("review_action")
            if action not in action_to_index:
                action_to_index[action] = current_index
                current_index += 1
            sentence["target_index"] = action_to_index[action]

    try:
        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        print(f"Added unique target indexes and saved to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")



def filter_review_sentences(input_file, output_file):
    """
    Extracts all 'review_sentences' from each object in the JSON and writes them
    into a single flattened list.

    Args:
        input_file (str): The input JSON file containing review sentence objects.
        output_file (str): The output JSON file where filtered data will be written.

    Returns:
        None
    """
    with open(input_file, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    review_sentences = []
    for item in data:
        review_sentences.extend(item.get("review_sentences", []))

    try:
        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(review_sentences, outfile, ensure_ascii=False, indent=4)
        print(f"Filtered review sentences saved to {output_file}")
    except Exception as e:
        print(f"Error writing {output_file}: {e}")


def filter_instances_by_review_action(input_file, output_folder, target_action):
    """
    Filters review sentences matching a specific 'review_action' value from a JSON file.

    Args:
        input_file (str): The path to the merged JSON file.
        output_folder (str): The directory to save the filtered output.
        target_action (str): The 'review_action' value to filter (e.g., "arg_request").

    Returns:
        None
    """
    filtered_instances = []

    try:
        with open(input_file, 'r', encoding='utf-8') as json_file:
            merged_data = json.load(json_file)
            for obj in merged_data:
                for sentence in obj.get("review_sentences", []):
                    if sentence.get("review_action") == target_action:
                        filtered_instances.append(sentence)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return

    output_file = os.path.join(output_folder, f"filtered_{target_action}.json")

    try:
        with open(output_file, 'w', encoding='utf-8') as json_output_file:
            json.dump(filtered_instances, json_output_file, ensure_ascii=False, indent=4)
        print(f"Filtered instances for action '{target_action}' saved to {output_file}")
    except Exception as e:
        print(f"Error writing {output_file}: {e}")



def convert_json_to_jsonl(input_file, output_folder, target_action):
    """
    Converts a JSON list file into a JSON Lines (.jsonl) file.

    Args:
        input_file (str): The path to the input JSON file (list of objects).
        output_folder (str): The directory path to store the resulting JSONL file.
        target_action (str): A label (e.g., "train" or "arg_request") to name the JSONL file.

    Returns:
        None
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

            if not isinstance(data, list):
                print(f"Data in {input_file} is not a list. Conversion to JSONL aborted.")
                return

            output_file = os.path.join(output_folder, f"filtered_{target_action}.jsonl")
            with open(output_file, 'w', encoding='utf-8') as jsonl_file:
                for item in data:
                    jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')

            print(f"Converted {input_file} to JSONL: {output_file}")
    except Exception as e:
        print(f"Error converting {input_file} to JSONL: {e}")



def add_index_to_jsonl(input_file, output_file):
    """
    Reads a JSONL file and adds a sequential 'index' field to each JSON object.

    Args:
        input_file (str): The path to the input JSONL file.
        output_file (str): The path for the output JSONL file with index fields.

    Returns:
        None
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:

            for i, line in enumerate(infile):
                obj = json.loads(line.strip())
                obj["index"] = i
                outfile.write(json.dumps(obj, ensure_ascii=False) + '\n')

        print(f"Added 'index' field to each entry in {input_file} -> {output_file}")
    except Exception as e:
        print(f"Error adding index to JSONL {input_file}: {e}")



if __name__ == "__main__":
    # Define file/folder paths
    folder_path = "backend/nlp/request_classifier/DISAPERE/final_dataset/test"
    output_folder = "backend/nlp/request_classifier/DISAPERE/final_dataset/"
    input_file = "backend/nlp/request_classifier/DISAPERE/final_dataset/merged_output.json"
    target_index = "backend/nlp/request_classifier/DISAPERE/final_dataset/merged_output_index.json"
    only_review_sentences = "backend/nlp/request_classifier/DISAPERE/final_dataset/only_review_sentences.json"
    filtered_file = "backend/nlp/request_classifier/DISAPERE/final_dataset/filtered_arg_request.json"
    request_file = "backend/nlp/request_classifier/DISAPERE/final_dataset/filtered_arg_request.jsonl"
    request_i__file = "backend/nlp/request_classifier/DISAPERE/final_dataset/filtered_arg_request_index.jsonl"

    # Merge JSON files into a single file
    merge_json_files(folder_path, input_file)

    # Add unique target indexes based on 'review_action'
    add_unique_target_index_to_nested_list(input_file, target_index)

    # Extract only the review sentences from the merged file
    filter_review_sentences(target_index, only_review_sentences)

    # Filter instances by a specific action (arg_request)
    filter_instances_by_review_action(input_file, output_folder, "arg_request")

    # Convert the full set of review sentences to JSONL
    convert_json_to_jsonl(only_review_sentences, output_folder, "train")

    # Add indexes to each line/object in the JSONL file
    add_index_to_jsonl(
        os.path.join(output_folder, "filtered_train.jsonl"),
        request_i__file
    )

    # Create a CSV with specific fields from the JSONL for further analysis
    filtered_data = []
    jsonl_input_path = os.path.join(output_folder, "filtered_train.jsonl")

    try:
        with open(jsonl_input_path, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line.strip())
                filtered_entry = {
                    "text": entry.get("text"),
                    "index": entry.get("index"),
                    "review_action": entry.get("review_action"),
                    "fine_review_action": entry.get("fine_review_action"),
                    "aspect": entry.get("aspect")
                }
                filtered_data.append(filtered_entry)

        df = pd.DataFrame(filtered_data)
        # Create a binary 'target' column: 1 if 'review_action' == 'arg_request', else 0
        df['target'] = df['review_action'].apply(lambda x: 1 if x == 'arg_request' else 0)

        csv_output_path = os.path.join(output_folder, "Request", "test.csv")
        os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
        df.to_csv(csv_output_path, index=False)
        print(f"CSV with filtered data saved to {csv_output_path}")

    except Exception as e:
        print(f"Error processing JSONL {jsonl_input_path}: {e}")
