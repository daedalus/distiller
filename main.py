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
from bitarray import bitarray
from transformers import AutoTokenizer, AutoModelForCausalLM
from urllib.parse import urlparse

sys.setrecursionlimit(10**6)


class BloomFilter:
    def __init__(self, power2=27, hash_count=8):
        # size = 2^power bits
        self.size = (1 << power2) - 1  # bitmask for indexing
        self.bit_array = bitarray(self.size + 1)  # allocate full array size
        self.bit_array.setall(0)
        self.hash_count = hash_count

    def _hashes(self, item):
        hashes = []
        prefix = hashlib.sha256(item.encode()).digest()
        for i in range(self.hash_count):
            i_bytes = str(i).encode()
            #h_bytes = hashlib.sha256(prefix + i_bytes).digest()
            idx = mmh3.hash(prefix + i_bytes, 0, signed=False) & self.size
            #idx = int.from_bytes(h_bytes, 'big') & self.size  # bitmask instead of modulo
            hashes.append(idx)
        return hashes

    def add(self, item):
        for idx in self._hashes(item):
            self.bit_array[idx] = 1

    def __contains__(self, item):
        return all(self.bit_array[idx] for idx in self._hashes(item))


class Distiller:
    def __init__(self, model_name, db_path, max_depth=10, compression_level=6, seed=None, bloom_size=27, bloom_hash_count=6, retrieve_to_bloom=False, use_unsloth=False, max_tokens=1024, api_url = None, api_key = None, system_prompt=None, max_ngrams=10):
        """
        Initialize the Distiller with model, database path, and other parameters.
        :param model_name: Name of the Huggingface model to use.
        :param db_path: Path to the SQLite database.
        :param max_depth: Maximum recursion depth for text generation.
        :param compression_level: Zlib compression level (1-9). 
        """
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
        self.max_ngrams = max_ngrams

        if api_url is None:
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    print(Fore.BLUE + f"[!] Cuda is available." + Style.RESET_ALL)
                    torch.cuda.manual_seed_all(seed)
                print(Fore.BLUE + f"[-] Torch seed set to {seed}" + Style.RESET_ALL)
        else:
            self.api_client = openai.OpenAI(base_url=self.api_url, api_key=self.api_key)
            print(Fore.BLUE +   f"[-] Using remote API: {api_url}" + Style.RESET_ALL)


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
        if self.retrieve_to_bloom:
            self.retrieve_to_bloom()

    def load_model(self):
        if self.use_unsloth:
            device = 'cuda'
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = self.model_name,
                dtype = None,
                load_in_4bit = True,
            )
            #max_seq_length = 32768,
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
        """
        Retrieve words from the database and add them to the Bloom filter.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT data FROM texts")
            rows = cursor.fetchall()
            for row in rows:
                if row[0]: self.bloom.add(zlib.decompress(row[0]).decode("utf8"))
            print(Fore.GREEN + "[+] Retrieved words from database to Bloom filter." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"[!] Error retrieving words from database: {e}" + Style.RESET_ALL)

    def sanitize(self, s):
        s = re.sub(r'\n+', '\n', s)
        s = re.sub(r' +', ' ', s)
        return s.strip()

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
        """Generates a commit message using OpenAI's GPT."""
        try:
            response = self.api_client.chat.completions.create(
            #response = self.api_client.completions.create(
                model=self.api_model,
                messages=[{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
            )
            self.retry_sleep = 1
            return response.choices[0].message.content, response.usage.completion_tokens

        except Exception as e:
            self.retry_sleep *= 2
            print(f"Error generating commit message: {e} retry in {self.retry_sleep}")
            time.sleep(self.retry_sleep)
            #self.retry_sleep = 1  
            return self.llm_api_generate(prompt)
          


    def generate(self, prompt):
        if self.api_url is None:
            return self.llm_local_generate(prompt)
        else:
            return self.llm_api_generate(prompt)

    def all_ngrams(self,text):
        text = text.replace("\n", "")
        words = text.split()
        if self.max_ngrams == -1:
            n = len(words)
        else:
            n = self.max_ngrams    
        for size in range(n,0,-1):
            for i in range(n - size + 1):
                yield " ".join(words[i:i + size])


    def g(self, words, depth=0):
        if depth > self.max_depth:
            return

        t0 = time.time()
        
        if depth == 0:
          text, tok = self.generate(words) 
          td = time.time() - t0
          yield text, tok,td, 0
          #yield from self.g(text, depth = 1)
        """
        LW = words.split('\n')
        if len(LW) > 1:
            for word in LW:
                try:
                    clean_word = self.sanitize(word)
                    text, tok = self.generate(clean_word)
                    clean_text = self.sanitize(text)
                    if clean_text not in self.bloom:
                        self.bloom.add(clean_text)
                        td = time.time() - t0
                        yield clean_text, tok, td, depth
                        yield from self.g(clean_text, depth=depth + 1)
                except Exception as e:
                    print(Fore.RED + f"[!] Generation error: {e}" + Style.RESET_ALL)
 
        for word in words.split():
            try:
                clean_word = self.sanitize(word)
                text, tok = self.generate(clean_word)
                clean_text = self.sanitize(text)
                if clean_text not in self.bloom:
                    self.bloom.add(clean_text)
                    td = time.time() - t0
                    yield clean_text, tok, td, depth
                yield from self.g(clean_text, depth=depth + 1)
            except Exception as e:
                print(Fore.RED + f"[!] Generation error: {e}" + Style.RESET_ALL)
        """
        for ngram in self.all_ngrams(words):
            try:
                #clean_word = self.sanitize(ngram)
                text, tok = self.generate(ngram)
                clean_text = self.sanitize(text)
                if clean_text not in self.bloom:
                    self.bloom.add(clean_text)
                    td = time.time() - t0
                    yield clean_text, tok, td, depth
                yield from self.g(clean_text, depth=depth + 1)
            except Exception as e:
                print(Fore.RED + f"[!] Generation error: {e}" + Style.RESET_ALL)


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
                    data = zlib.compress(text.encode("utf8"), level=self.compression_level)
                    self.conn.execute(
                        "INSERT INTO texts (seed, model, word, data, tok) VALUES (?,?, ?, ?,?)",
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


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Distiller with Bloom filter and SQLite storage.")
    parser.add_argument("prompt", help="Root word or prompt to distill.")
    parser.add_argument("--model", default="distilgpt2", help="Huggingface model name (default: distilgpt2).")
    parser.add_argument("--db", default="words/data.db", help="Path to SQLite database (default: words/data.db).")
    parser.add_argument("--max-depth", type=int, default=10, help="Max recursion depth (default: 10).")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens (default: 1024).")
    parser.add_argument("--compression-level", type=int, choices=range(1, 10), default=6, help="Zlib compression level (1-9, default: 6).")
    parser.add_argument("--seed", type=int, help="Torch manual seed (optional).")
    parser.add_argument("--bloom-size", type=int, default=26, help="Bloom filter size (default: 100,000,000).")
    parser.add_argument("--bloom-hash-count", type=int, default=6, help="Bloom filter hash count (default: 6).")
    parser.add_argument("--max-ngrams", type=int, default=10, help="Max ngrams (default: 10).")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output.")
    parser.add_argument("--retrieve-to-bloom", action="store_true", help="Retrieve words from the database to the Bloom filter.")
    parser.add_argument("--use-unsloth", action="store_true", help="Use unsloth")
    parser.add_argument("--api-url", default=None, help="OpenAI compatible API url.")
    parser.add_argument("--api-key", default=None, help="API key for auth.")
    parser.add_argument("--system-prompt", default='You are a helpful AI assistant.', help="System prompt")
    parser.add_argument("--threads", type=int, default=None, help="Number of CPU threads for PyTorch (default: auto)")
    parser.add_argument("--secrets-file", default=None, help="Specify the secrets json file.")


    if parser.parse_args().no_color:
        Fore.RESET = ""
        Style.RESET_ALL = ""
    return parser.parse_args()

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
        use_unsloth = args.use_unsloth and UNSLOTH_AVAILABLE,
        api_url = args.api_url,
        api_key = api_key,
        max_tokens = args.max_tokens,
        system_prompt = args.system_prompt,
        max_ngrams = args.max_ngrams,
    )
    distiller.distill(args.prompt)

