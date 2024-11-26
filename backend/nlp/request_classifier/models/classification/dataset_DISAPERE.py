import os
import json
from collections import defaultdict
import pandas as pd

folder_path = "backend/nlp/request_classifier/DISAPERE/final_dataset/train"  
output_folder = "backend/nlp/request_classifier/DISAPERE/final_dataset/"    
input_file = "backend/nlp/request_classifier/DISAPERE/final_dataset/merged_output.json"    
target_index = "backend/nlp/request_classifier/DISAPERE/final_dataset/merged_output_index.json"   
only_review_sentences = "backend/nlp/request_classifier/DISAPERE/final_dataset/only_review_sentences.json"   
filtered_file = "backend/nlp/request_classifier/DISAPERE/final_dataset/filtered_arg_request.json"    
request_file = "backend/nlp/request_classifier/DISAPERE/final_dataset/filtered_arg_request.jsonl"    
request_i__file = "backend/nlp/request_classifier/DISAPERE/final_dataset/filtered_arg_request_index.jsonl"   

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
filter_instances_by_review_action(input_file, output_folder,"arg_request")
convert_json_to_jsonl(only_review_sentences, output_folder,"train")
add_index_to_jsonl("backend/nlp/request_classifier/DISAPERE/final_dataset/filtered_train.jsonl", request_i__file)


data = []
filtered_data = []

# JSONL-Datei öffnen und nur die gewünschten Felder extrahieren
with open("backend/nlp/request_classifier/DISAPERE/final_dataset/filtered_train.jsonl", 'r', encoding='utf-8') as file:
    for line in file:
        entry = json.loads(line.strip())  # JSON-Objekt laden
        # Nur die gewünschten Felder extrahieren
        filtered_entry = {
            "text": entry.get("text"),
            "index": entry.get("index"),
            "review_action": entry.get("review_action"),
            "fine_review_action": entry.get("fine_review_action"),
            "aspect": entry.get("aspect")
        }
        filtered_data.append(filtered_entry)

df = pd.DataFrame(filtered_data)
#df['target'] = df['review_action'].apply(lambda x: 1 if x == 'arg_request' else 0)

#df.to_csv("backend/request_classifier/DISAPERE/final_dataset/Request/test.csv", index=False)

#df = df[df["review_action"] == "arg_request"]
unique_labels = df["aspect"].unique()

# Ein Mapping für jedes Label erstellen
label_to_value = {label: idx for idx, label in enumerate(unique_labels)}

# Neue Spalte mit den zugewiesenen Werten hinzufügen
df["target"] = df["aspect"].map(label_to_value)

unique_labels = df["target"].unique()
print("Einzigartige Labels:", unique_labels)
df.to_csv("backend/nlp/request_classifier/DISAPERE/final_dataset/fine_request/train_attitude.csv", index=False)

