

import re
import random
import zlib
import sys
import hashlib
import sqlite3
import torch
import time
from colorama import Fore, Style
from bitarray import bitarray
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.setrecursionlimit(10**6)



def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device

#@title BLOOMFILTER

# ========== Bloom filter ==========


class BloomFilter:
    def __init__(self, size=1_000_000, hash_count=5):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def _hashes(self, item):
        hashes = []
        for i in range(self.hash_count):
            h = hashlib.sha256((str(i) + item).encode()).hexdigest()
            idx = int(h, 16) % self.size
            hashes.append(idx)
        return hashes

    def add(self, item):
        for idx in self._hashes(item):
            self.bit_array[idx] = 1

    def __contains__(self, item):
        return all(self.bit_array[idx] for idx in self._hashes(item))

bloom1 = BloomFilter(size=1_00_000_000, hash_count=6)



# Precompile emoji pattern
emoji_pattern = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"  # Flags
    "\U0001F300-\U0001F5FF"  # Symbols & pictographs
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F680-\U0001F6FF"  # Transport & map symbols
    "\U0001F700-\U0001F77F"  # Alchemical
    "\U0001F780-\U0001F7FF"  # Geometric extended
    "\U0001F800-\U0001F8FF"  # Supplemental arrows
    "\U0001F900-\U0001F9FF"  # Supplemental pictographs
    "\U0001FA00-\U0001FA6F"  # Chess etc
    "\U0001FA70-\U0001FAFF"  # Extended pictographs
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed
    "\U00002600-\U000026FF"  # Misc symbols
    "\U00002300-\U000023FF"  # Misc technical
    "\u200D"                 # Zero width joiner
    "\uFE0F"                 # VS-16
    "]+",
    flags=re.UNICODE
)

def build_bad_token_ids(tokenizer):
    # Build list of bad token IDs
    bad_token_ids = []
    for i in range(tokenizer.vocab_size):
        decoded = tokenizer.decode([i])
        if emoji_pattern.search(decoded):
            bad_token_ids.append([i])
    return bad_token_ids

def llm_generate(prompt, model, tokenizer, device, bad_token_ids):
    # Tokenizar
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generar texto sin emojis
    outputs = model.generate(
        **inputs,
        max_length=1024,
        do_sample=True,
        top_k=50,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        bad_words_ids=bad_token_ids
    )
    
    # Decodificar y mostrar
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    tokens = outputs.shape[1]
    return generated_text, tokens




def sanitize(str):
  return str.replace("\n\n\n\n","")
 
TOK = 0


def g(words, model, tokenizer, device, bad_token_ids, depth=0):
    t0 = time.time()
    if words not in bloom1:
      bloom1.add(words)
      for word in words.split(" "):
        text, tok = llm_generate(sanitize(word), model, tokenizer, device, bad_token_ids)
        td = time.time() - t0 
        yield sanitize(text), tok, td, depth
        yield from g(text, model, tokenizer, device, bad_token_ids, depth = depth + 1)


def distill(word, model, tokenizer, device, bad_token_ids):
  n =  0
  global TOK
  for text,tok,td,depth in g(word, model, tokenizer, device, bad_token_ids ):
      TOK += tok
      ts = tok/td
      print(Fore.YELLOW + f"[+] Generation: {n}, depth: {depth}, tokens: {tok} at {round(td,2)} tokens/s \n" + Fore.GREEN  +   f"[{text}]" + Style.RESET_ALL   )
      data = zlib.compress(text.encode("utf8"))
      conn.execute("INSERT INTO texts (word, data, tok) VALUES (?,?,?)", (word,data,tok,))
      conn.commit()

      print(f"[+] Total tokens: {TOK}")
      #f.flush()
      n+=1

if __name__ == '__main__':
    if len(sys.argv) > 1:
        conn = sqlite3.connect("words/data.db")
        conn.execute("CREATE TABLE IF NOT EXISTS texts (id INTEGER PRIMARY KEY, word TEXT, data BLOB, tok INT)")

        model_name = "distilgpt2"
        #model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        #model_name = "sgugger/rwkv-430M-pile"
        #model_name = "microsoft/bitnet-b1.58-2B-4T"
   
        model, tokenizer, device = load_model(model_name)
        print(Fore.BLUE +  f"[!] Using model: {model_name} in device: {device}." + Style.RESET_ALL)

        bad_token_ids = build_bad_token_ids(tokenizer)

        distill(sys.argv[1], model, tokenizer, device, bad_token_ids) 
