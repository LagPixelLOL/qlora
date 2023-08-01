import json
import sys

def load_json_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def save_jsonl_file(data, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            for item in data:
                json.dump(item, file, ensure_ascii=False)
                file.write('\n')
    except Exception as e:
        print(f"Error saving file: {e}")
        sys.exit(1)

def main():
    input_filename = input("Enter the input filename: ")
    input_filename += ".json"

    json_data = load_json_file(input_filename)

    if not isinstance(json_data, list):
        print("Error: JSON data is not an array.")
        sys.exit(1)

    jsonl_data = []

    for entry in json_data:
        dialog = entry.get("conversations", [])
        if len(dialog) == 2:
            i_text = dialog[0].get("value", "")
            r_text = dialog[1].get("value", "")
            formatted_entry = {
                "input": f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{i_text}\n\n### Response:",
                "output": f" {r_text}"
            }
            jsonl_data.append(formatted_entry)

    output_filename = input("Enter the output filename: ")
    output_filename += ".jsonl"

    save_jsonl_file(jsonl_data, output_filename)

    print("Conversion completed.")

if __name__ == '__main__':
    main()
