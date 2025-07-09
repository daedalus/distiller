import sys
sys.setrecursionlimit(10**6)
# Importar
from transformers import AutoTokenizer, AutoModelForCausalLM

# Modelo pequeño (puedes cambiar a 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' si quieres algo más potente)
model_name = "distilgpt2"

# Cargar modelo y tokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)
tokenizer.pad_token = tokenizer.eos_token

#@title BLOOMFILTER

# ========== Bloom filter ==========
from bitarray import bitarray
import hashlib

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

#@title EXTRACTION
 
def llm_generate(prompt):
  # Tokenizar
  inputs = tokenizer(prompt, return_tensors="pt")
  # Generar texto
  outputs = model.generate(**inputs, max_length=100, do_sample=True, top_k=50, pad_token_id=tokenizer.eos_token_id, max_new_tokens=200)
  # Decodificar y mostrar
  generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
  #print(generated_text)


  total_tokens = outputs.shape[1]
  #new_tokens = total_tokens - prompt_len

  return generated_text, total_tokens

bloom = BloomFilter(size=1_000_000_000, hash_count=6)
 
TOK = 0

def g(words, depth=0):
    global TOK
    """
    if words == "1":
      w = llm_generate("1")
      bloom.add(w)
      yield w
    """
    if words not in bloom:
      text, tok = llm_generate(words)
      TOK += tok
      yield text
      bloom.add(words)
      for word in words.split(" "):
        text, tok = llm_generate(word.replace("\n\n\n\n",""))
        TOK += tok
        yield text.replace("\n\n\n\n","")
        yield from g(t, depth = depth + 1)



def gwords(word):
  n =  0
  global TOK
  with open(f"words/words_{word}.txt", "a") as f:
    for words in g(word):
      words = words.replace("\n\n\n\n","")
      print(f"Generation: {n}, total tokens: {TOK} || [{words}]")
      f.write(words+"\n")
      f.flush()
      n+=1


def loadwords(filename):
  with open(filename,"r") as f:
      return f.readlines()
      

if len(sys.argv) > 1:
  gwords(sys.argv[1]) 
else:
  WORD_LIST += loadwords("/usr/share/dict/american-english-insane")
  L = len(WORD_LIST)
  for word in WORD_LIST:
    print("New topic:", word)
    gwords(word.strip())
