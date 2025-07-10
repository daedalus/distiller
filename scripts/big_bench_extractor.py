import os
import json
import sys

def process_json_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "examples" in data and isinstance(data["examples"], list):
                for i, example in enumerate(data["examples"]):
                    if isinstance(example, dict) and "input" in example:
                        sys.stderr.write(f"{filepath} [example {i}] input:\n")
                        text = example['input']
                        if len(text.split("\n")) > 1:
                            text = text.replace("\n","\\n")
                        print(text)
    except Exception as e:
        print(f"Failed to process {filepath}: {e}")

def walk_and_process_jsons(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                filepath = os.path.join(dirpath, filename)
                process_json_file(filepath)

# Change this to your target directory
target_directory = sys.argv[1]
walk_and_process_jsons(target_directory)

