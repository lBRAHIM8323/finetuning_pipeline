import os
import json

def combine_jsons_in_folder(folder_path):
    combined = []
    
    # List all .json files except the final output file
    for filename in os.listdir(folder_path):
        if filename.endswith(".json") and filename != "Finetuning_dataset.json":
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        combined.extend(data)
                    elif isinstance(data, dict):
                        combined.append(data)
                    else:
                        print(f"Skipping {filename}: Unsupported JSON format.")
                except json.JSONDecodeError as e:
                    print(f"Error reading {filename}: {e}")
    
    # Save the combined data
    output_file = os.path.join(folder_path, "Finetuning_dataset.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    print(f"Combined JSON saved to: {output_file} ({len(combined)} entries)")

# Example usage:
combine_jsons_in_folder("/home/hirdesh/Desktop/JSON_comp")
