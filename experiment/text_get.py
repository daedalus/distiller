import sqlite3
import zlib
import zstd

import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
lm_model = GPT2LMHeadModel.from_pretrained("gpt2")
lm_model.eval()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def split_into_windows(text, window_size=100):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + window_size] for i in range(0, len(tokens), window_size)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def compute_perplexity(text):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = lm_model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()

def embedding_similarity(prompt, chunk):
    emb_prompt = embedding_model.encode(prompt, convert_to_tensor=True)
    emb_chunk = embedding_model.encode(chunk, convert_to_tensor=True)
    return util.cos_sim(emb_prompt, emb_chunk).item()

def penalized_perplexity(perplexities, penalty_factor=2.0):
    mean_ppl = np.mean(perplexities)
    std_ppl = np.std(perplexities)
    return round(mean_ppl + penalty_factor * std_ppl, 2)

def evaluate_generation(prompt, generation, window_size=100, penalty_factor=2.0):
    chunks = split_into_windows(generation, window_size)
    results = []
    perplexities = []
    similarities = []

    for i, chunk in enumerate(chunks):
        ppl = compute_perplexity(chunk)
        sim = embedding_similarity(prompt, chunk)
        perplexities.append(ppl)
        similarities.append(sim)
        results.append({
            "window": i,
            "text": chunk[:100] + "...",
            "perplexity": round(ppl, 2),
            "similarity": round(sim, 4)
        })

    final_score = {
        #"results": results,
        "penalized_perplexity": penalized_perplexity(perplexities, penalty_factor),
        "max_perplexity": round(np.max(perplexities), 4),
        "avg_perplexity": round(np.mean(perplexities), 2),
        "std_perplexity": round(np.std(perplexities), 2),
        "min_perplexity": round(np.min(perplexities), 2),
        "max_similarity": round(np.max(similarities), 4),
        "avg_similarity": round(np.mean(similarities), 4),
        "std_similarity": round(np.std(similarities), 4),
        "min_similarity": round(np.min(similarities), 4)
    }

    return {"windows": results, "summary": final_score}



db_path = "words/data.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("select word, data, compression_algo from texts;")
for row in cursor.fetchall():
  prompt = row[0]
  if row[2] == "zlib":
    text = zlib.decompress(row[1])
  if row[2] == "zstd":
    text = zstd.decompress(row[1])
  text = text.decode("utf8")

  print("-"*60)
  R = evaluate_generation(prompt, text, window_size=50)
  print(R['summary']['penalized_perplexity'])
  #print(tokenizer.encode(text))
    
