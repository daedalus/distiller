import re
import zlib
import sys
import hashlib
import sqlite3
import torch
import time
import argparse
from colorama import Fore, Style
from bitarray import bitarray
from transformers import AutoTokenizer, AutoModelForCausalLM

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
            h_bytes = hashlib.sha256(prefix + i_bytes).digest()
            idx = int.from_bytes(h_bytes, 'big') & self.size  # bitmask instead of modulo
            hashes.append(idx)
        return hashes

    def add(self, item):
        for idx in self._hashes(item):
            self.bit_array[idx] = 1

    def __contains__(self, item):
        return all(self.bit_array[idx] for idx in self._hashes(item))


class Distiller:
    def __init__(self, model_name, db_path, max_depth=10, compression_level=6, seed=None, bloom_size=27, bloom_hash_count=6, retrieve_to_bloom=False):
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

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                print(Fore.BLUE + f"[!] Cuda is available." + Style.RESET_ALL)
                torch.cuda.manual_seed_all(seed)
            print(Fore.BLUE + f"[!] Torch seed set to {seed}" + Style.RESET_ALL)

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

        self.model, self.tokenizer, self.device = self.load_model()
        self.bad_token_ids = self.build_bad_token_ids()
        self.conn = self.init_db()
        if self.retrieve_to_bloom:
            self.retrieve_to_bloom()

    def load_model(self):
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
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS texts (
                id INTEGER PRIMARY KEY,
                word TEXT,
                data BLOB NOT NULL UNIQUE,
                tok INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
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

    def llm_generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=1024,
            do_sample=True,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id,
            bad_words_ids=self.bad_token_ids
        )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens = outputs.shape[1]
        return generated_text, tokens

    def g(self, words, depth=0):
        if depth > self.max_depth:
            return

        t0 = time.time()
        for word in words.split():
            try:
                clean_word = self.sanitize(word)
                text, tok = self.llm_generate(clean_word)
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
                        "INSERT INTO texts (word, data, tok) VALUES (?, ?, ?)",
                        (root_word, data, tok)
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
    parser.add_argument("--compression-level", type=int, choices=range(1, 10), default=6, help="Zlib compression level (1-9, default: 6).")
    parser.add_argument("--seed", type=int, help="Torch manual seed (optional).")

    parser.add_argument("--bloom-size", type=int, default=27, help="Bloom filter size (default: 100,000,000).")
    parser.add_argument("--bloom-hash-count", type=int, default=6, help="Bloom filter hash count (default: 6).")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output.")
    parser.add_argument("--retrieve-to-bloom", action="store_true", help="Retrieve words from the database to the Bloom filter.")


    if parser.parse_args().no_color:
        Fore.RESET = ""
        Style.RESET_ALL = ""
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    distiller = Distiller(
        model_name=args.model,
        db_path=args.db,
        max_depth=args.max_depth,
        compression_level=args.compression_level,
        seed=args.seed,
        bloom_size=args.bloom_size, 
        bloom_hash_count=args.bloom_hash_count,
        retrieve_to_bloom=args.retrieve_to_bloom
    )
    distiller.distill(args.prompt)

