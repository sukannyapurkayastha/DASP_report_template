import os
import json
from collections import defaultdict


folder_path = "backend/classificationmodel/DISAPERE/final_dataset/train"  
output_folder = "backend/classificationmodel/DISAPERE/final_dataset/"    
input_file = "backend/classificationmodel/DISAPERE/final_dataset/merged_output.json"    
target_index = "backend/classificationmodel/DISAPERE/final_dataset/merged_output_index.json"   
only_review_sentences = "backend/classificationmodel/DISAPERE/final_dataset/only_review_sentences.json"   
filtered_file = "backend/classificationmodel/DISAPERE/final_dataset/filtered_arg_request.json"    
request_file = "backend/classificationmodel/DISAPERE/final_dataset/filtered_arg_request.jsonl"    
request_i__file = "backend/classificationmodel/DISAPERE/final_dataset/filtered_arg_request_index.jsonl"   

def merge_json_files(folder_path, output_file):

    merged_data = []

    for file_name in os.listdir(folder_path):

            file_path = os.path.join(folder_path, file_name)

            try:
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    merged_data.append(data)
            except Exception as e:
                print(f"Fehler beim Lesen von {file_name}: {e}")

    try:
        with open(output_file, 'w', encoding='utf-8') as output_json_file:
            json.dump(merged_data, output_json_file, ensure_ascii=False, indent=4)
        print(f"Zusammengeführte JSON-Datei wurde erstellt: {output_file}")
    except Exception as e:
        print(f"Fehler beim Schreiben der Datei {output_file}: {e}")



def filter_instances_by_review_action(input_file, output_folder, target_action):

    filtered_instances = []

    try:
        with open(input_file, 'r', encoding='utf-8') as json_file:
            merged_data = json.load(json_file)

            for obj in merged_data:
                if "review_sentences" in obj:
                    for sentence in obj["review_sentences"]:
                        if sentence.get("review_action") == target_action:
                            filtered_instances.append(sentence)
    except Exception as e:
        print(f"Fehler beim Lesen der Datei {input_file}: {e}")
        return

    output_file = os.path.join(output_folder, f"filtered_{target_action}.json")

    try:
        with open(output_file, 'w', encoding='utf-8') as json_output_file:
            json.dump(filtered_instances, json_output_file, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"Fehler beim Schreiben der Datei {output_file}: {e}")

def convert_json_to_jsonl(input_file, output_folder, target_action):

    try:
       
        with open(input_file, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

            if not isinstance(data, list):
        
                return
            
            output_file = os.path.join(output_folder, f"filtered_{target_action}.jsonl")
            # JSONL-Datei schreiben
            with open(output_file, 'w', encoding='utf-8') as jsonl_file:
                for item in data:
                    print(item)
                    jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')

    except Exception as e:
        print(f"Fehler beim Konvertieren der Datei: {e}")


def add_index_to_jsonl(input_file, output_file):

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for i, line in enumerate(infile):
                obj = json.loads(line.strip())  # JSON-Objekt aus der Zeile laden
                obj["index"] = i  # Index hinzufügen
                outfile.write(json.dumps(obj, ensure_ascii=False) + '\n')  # Neue Zeile schreiben
    except Exception as e:
        print(f"Fehler beim Hinzufügen des Index: {e}")


def add_unique_target_index_to_nested_list(input_file, output_file):
   
    with open(input_file, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    action_to_index = {}
    current_index = 0

    for item in data:
        for sentence in item["review_sentences"]:
            action = sentence["review_action"]
            if action not in action_to_index:
                action_to_index[action] = current_index
                current_index += 1
 
            sentence["target_index"] = action_to_index[action]

    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

def filter_review_sentences(input_file, output_file):

    with open(input_file, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    review_sentences = []
    for item in data:
        if "review_sentences" in item:
            review_sentences.extend(item["review_sentences"])

    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(review_sentences, outfile, ensure_ascii=False, indent=4)
        


merge_json_files(folder_path, input_file)
add_unique_target_index_to_nested_list(input_file, target_index)
filter_review_sentences(target_index, only_review_sentences)
#filter_instances_by_review_action(input_file, output_folder,"arg_request")
convert_json_to_jsonl(only_review_sentences, output_folder,"train")
add_index_to_jsonl("backend/classificationmodel/DISAPERE/final_dataset/filtered_train.jsonl", request_i__file)
