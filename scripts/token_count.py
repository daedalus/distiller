import os

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")  # or another model

def count_tokens_in_directory(directory):
    total_tokens = 0
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            print(f"Processing {filepath}...",end="")
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            except UnicodeDecodeError:
                # Optional: try fallback encoding
                with open(filepath, 'r', encoding='latin-1') as f:
                    text = f.read()
            tokens = tokenizer.encode(text)
            lt = len(tokens)
            total_tokens += lt
            print(f"tokens: {lt} total tokens: {total_tokens}")
            del text
            del tokens

    return total_tokens

import sys

print(count_tokens_in_directory(sys.argv[1]))
