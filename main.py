from colorama import Fore, Style
try:
    import unsloth
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    print(Fore.BLUE + "[-] Unsloth available" + Style.RESET_ALL)
except:
    UNSLOTH_AVAILABLE = False
    print(Fore.RED + "[!] Unsloth unavailable"+ Style.RESET_ALL)
import os
import re
import zlib
import sys
import hashlib
import sqlite3
import torch
import time
import argparse
import openai
import mmh3
import json
from typing import List, Tuple, Generator
from bitarray import bitarray
from transformers import AutoTokenizer, AutoModelForCausalLM
from urllib.parse import urlparse
from huggingface_hub import InferenceClient
from lib.bloomfilter import BloomFilter
from lib.utils import sanitize, compress, parse_args
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Generator, Dict
from joblib import Memory

class Distiller:
    def __init__(self, model_name, db_path, max_depth=10, compression_level=6, seed=None, 
                 bloom_size=27, bloom_hash_count=6, retrieve_to_bloom=False, use_unsloth=False, 
                 max_tokens=1024, api_url=None, api_key=None, system_prompt=None, 
                 max_ngrams=10, min_ngrams=0, api_hf_provider=None, prompt_prefixes=None,
                 batch_size=1, min_tfidf_score=0.2, max_ngram_length=4, tfidf_cache_dir=None):

        # Initialize core parameters
        self.model_name = model_name
        self.db_path = db_path
        self.max_depth = max_depth
        self.compression_level = compression_level
        self.bloom = BloomFilter(power2=bloom_size, hash_count=bloom_hash_count)
        self.TOK = 0
        self.seed = seed
        self.use_unsloth = use_unsloth
        self.max_tokens = max_tokens
        self.api_url = api_url
        self.api_key = api_key
        self.api_model = model_name
        self.retry_sleep = 1
        self.system_prompt = system_prompt
        self.min_ngrams = min_ngrams
        self.max_ngrams = max_ngrams
        self.batch_size = batch_size
        self.min_tfidf_score = min_tfidf_score
        self.max_ngram_length = max_ngram_length
        self.api_hf_provider = api_hf_provider
        self.prompt_prefixes = prompt_prefixes if prompt_prefixes is not None else ['']
         
        # Initialize TF-IDF components
        self.tfidf_memory = Memory(tfidf_cache_dir, verbose=0) if tfidf_cache_dir else None
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, max_ngram_length),
            stop_words='english'
        )
        if self.tfidf_memory:
            self.tfidf_vectorizer = self.tfidf_memory.cache(self.tfidf_vectorizer)
        self.corpus = []

        if api_url is None:
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    print(Fore.BLUE + f"[!] Cuda is available." + Style.RESET_ALL)
                    torch.cuda.manual_seed_all(seed)
                print(Fore.BLUE + f"[-] Torch seed set to {seed}" + Style.RESET_ALL)
        else:
            if api_hf_provider is None:
                self.api_client = openai.OpenAI(base_url=self.api_url, api_key=self.api_key)
                print(Fore.BLUE + f"[-] Using OpenAI remote API: {api_url}" + Style.RESET_ALL)
            else:
                self.api_client = InferenceClient(provider=self.api_hf_provider, api_key=self.api_key)

        self.emoji_pattern = re.compile(
            "["
            "\U0001F1E6-\U0001F1FF"
            "\U0001F300-\U0001F5FF"
            "\U0001F600-\U0001F64F"
            "\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FAFF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U00002600-\U000026FF"
            "\U00002300-\U000023FF"
            "\u200D"
            "\uFE0F"
            "]+", flags=re.UNICODE
        )

        if api_url is None:
            self.model, self.tokenizer, self.device = self.load_model()
            self.bad_token_ids = self.build_bad_token_ids()

        self.conn = self.init_db()
        if retrieve_to_bloom:
            self.retrieve_to_bloom()


    def load_model(self):
        if self.use_unsloth:
            device = 'cuda'
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
        print(Fore.BLUE + f"[!] Using model: {self.model_name} on device: {device}, with database: {self.db_path}." + Style.RESET_ALL)
        return model, tokenizer, device


    def build_bad_token_ids(self):
        bad_token_ids = []
        for i in range(self.tokenizer.vocab_size):
            decoded = self.tokenizer.decode([i])
            if self.emoji_pattern.search(decoded):
                bad_token_ids.append([i])
        return bad_token_ids


    def init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS texts (
                id INTEGER PRIMARY KEY,
                seed INT,
                model TEXT,
                word TEXT,
                data BLOB NOT NULL UNIQUE,
                tok INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
        cursor = conn.cursor()
        cursor.execute("select sum(tok) from texts;")
        tok = cursor.fetchone()[0]
        print(Fore.BLUE + f"[!] Total accumulated tokens in the database {tok}." + Style.RESET_ALL)
        return conn
    

    def retrieve_to_bloom(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT data FROM texts")
            rows = cursor.fetchall()
            for row in rows:
                if row[0]: self.bloom.add(zlib.decompress(row[0]).decode("utf8"))
            print(Fore.GREEN + "[+] Retrieved words from database to Bloom filter." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"[!] Error retrieving words from database: {e}" + Style.RESET_ALL)


    def llm_local_generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=self.max_tokens,
            do_sample=True,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id,
            bad_words_ids=self.bad_token_ids
        )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens = outputs.shape[1]
        return generated_text, tokens


    def llm_local_generate_batch(self, prompts: List[str]) -> Tuple[List[str], List[int]]:
        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=self.bad_token_ids
            )
            return [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ], [len(output) for output in outputs]
        except torch.cuda.OutOfMemoryError:
            self.batch_size = max(1, self.batch_size // 2)
            print(Fore.YELLOW + f"[!] Reduced batch size to {self.batch_size} due to OOM" + Style.RESET_ALL)
            return self.llm_local_generate_batch(prompts)


    def sleep_unlock(self):
        if self.retry_sleep > 0:
            print(Fore.BLUE + f"[-] Going to sleep for {self.retry_sleep} seconds to avoid overwelming the servers." + Style.RESET_ALL)
            time.sleep(self.retry_sleep)
            self.retty_sleep = 0


    def sleep_set(self):
        if self.retry_sleep == 0:
            self.retty_sleep = 1
        else:
            self.retty_sleep *= 2


    def llm_api_generate(self, prompt):
        try:
            response = self.api_client.chat.completions.create(
                model=self.api_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
            )
            self.retry_sleep = 1
            return response.choices[0].message.content, response.usage.completion_tokens
        except Exception as e:
            self.retry_sleep *= 2
            print(f"Error generating commit message: {e} retry in {self.retry_sleep}")
            time.sleep(self.retry_sleep)
            return self.llm_api_generate(prompt)


    def llm_api_generate_batch(self, prompts: List[str]) -> Tuple[List[str], List[int]]:
        texts, toks = [], []
        for prompt in prompts:
            text, tok = self.llm_api_generate(prompt)
            texts.append(text)
            toks.append(tok)
        return texts, toks


    def generate(self, prompt):
        if self.api_url is None:
            return self.llm_local_generate(prompt)
        else:
            return self.llm_api_generate(prompt)


    def generate_batch(self, prompts: List[str]) -> Tuple[List[str], List[int]]:
        if self.api_url is None:
            return self.llm_local_generate_batch(prompts)
        else:
            return self.llm_api_generate_batch(prompts)


    def get_tfidf_scores(self, text: str) -> Dict[str, float]:
        """Calculate TF-IDF scores for all n-grams in text"""
        self.corpus.append(text)
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            scores = zip(feature_names, tfidf_matrix[-1].toarray().flatten())
            return {ngram: score for ngram, score in scores if score > self.min_tfidf_score}
        except ValueError:
            return {}


    def all_ngrams(self, text: str) -> Generator[str, None, None]:
        """Yield n-grams in order of their TF-IDF importance"""
        text = text.replace("\n", "")
        tfidf_scores = self.get_tfidf_scores(text)
        
        # Get all possible ngrams
        words = text.split()
        all_ngrams = set()
        for n in range(min(self.max_ngrams, len(words)), self.min_ngrams, -1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i+n])
                all_ngrams.add(ngram)
        
        # Score and filter ngrams
        scored_ngrams = [(ngram, tfidf_scores.get(ngram, 0)) 
                        for ngram in all_ngrams]
        scored_ngrams.sort(key=lambda x: x[1], reverse=True)
        
        # Yield ngrams above threshold first, then others if needed
        for ngram, score in scored_ngrams:
            if score >= self.min_tfidf_score or len(scored_ngrams) < 5:  # Keep at least 5 ngrams
                yield ngram
   

    def g(self, root_prompt) -> Generator[Tuple[str, int, float, int], None, None]:
        stack = [(root_prompt, 0)]
    
        while stack:
            batch = []
            while len(batch) < self.batch_size and stack:
                prompt, depth = stack.pop()
                batch.append((prompt, depth))
            
            if not batch:
                continue
                
            prompts = [item[0] for item in batch]
            depths = [item[1] for item in batch]
            
            t0 = time.time()
            texts, toks = self.generate_batch(prompts)
            td = time.time() - t0
            
            for text, tok, depth in zip(texts, toks, depths):
                clean_text = sanitize(text)
                #clean_text = clean_text.replace(prompt, '')
                yield clean_text, tok, td, depth
                
                new_prompts = []
                for ngram in all_ngrams(clean_text):
                    for prefix in self.prompt_prefixes:
                        new_prompt = f"{prefix} {ngram}" if prefix else ngram
                    
                        if new_prompt in self.bloom:
                            continue
                        
                        self.bloom.add(new_prompt)
                        if depth + 1 < self.max_depth:
                            new_prompts.append((new_prompt, depth + 1))
            
                stack.extend(reversed(new_prompts))


    def distill(self, root_word):
        t0 = time.time()
        n = 0
        try:
            with self.conn:
                for text, tok, td, depth in self.g(root_word):
                    self.TOK += tok
                    ts = tok / td if td > 0 else 0
                    print(Fore.YELLOW + f"[+] Generation: {n}, depth: {depth}, tokens: {tok} at {round(ts, 2)} tokens/s\n" +
                          Fore.GREEN + f"[{text}]" + Style.RESET_ALL)
                    data = compress(text, level=self.compression_level)
                    self.conn.execute(
                        "INSERT or ignore INTO texts (seed, model, word, data, tok) VALUES (?,?, ?, ?,?)",
                        (self.seed, self.model_name, root_word, data, tok)
                    )
                    self.conn.commit()
                    lt, lc = len(text), len(data)
                    print(Fore.BLUE + f"[+] Text size: {lt} bytes, Compressed data size: {lc} bytes, ratio: {round(lt/lc,2)}" + Style.RESET_ALL)
                    
                    tdt = time.time() - t0
                    print(f"[+] Total tokens: {self.TOK}, total elapsed time: {round(tdt, 2)} seconds, total tokens/s: {round(self.TOK/tdt, 2)}")
                    n += 1
        except Exception as e:
            print(Fore.RED + f"[!] DB error: {e}" + Style.RESET_ALL)

if __name__ == '__main__':
    args = parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)
        print(Fore.BLUE + f"[-] Torch set to use {args.threads} CPU threads" + Style.RESET_ALL)

    if args.secrets_file is not None:
        parsed_url = urlparse(args.api_url)
        api_key = json.load(open(args.secrets_file,"r"))[parsed_url.hostname]
    else:
        api_key = args.api_key

    distiller = Distiller(
        model_name=args.model,
        db_path=args.db,
        max_depth=args.max_depth,
        compression_level=args.compression_level,
        seed=args.seed,
        bloom_size=args.bloom_size, 
        bloom_hash_count=args.bloom_hash_count,
        retrieve_to_bloom=args.retrieve_to_bloom,
        use_unsloth=args.use_unsloth and UNSLOTH_AVAILABLE,
        api_url=args.api_url,
        api_key=api_key,
        max_tokens=args.max_tokens,
        system_prompt=args.system_prompt,
        max_ngrams=args.max_ngrams,
        api_hf_provider=args.api_hf_provider,
        prompt_prefixes=args.prompt_prefixes,
        batch_size=args.batch_size,
    )
    distiller.distill(args.prompt)