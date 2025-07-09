import re
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


class BloomFilter:
    def __init__(self, size=100_000_000, hash_count=6):
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


class Distiller:
    def __init__(self, model_name="distilgpt2", db_path="words/data.db", max_depth=10):
        self.model_name = model_name
        self.db_path = db_path
        self.max_depth = max_depth
        self.bloom = BloomFilter()
        self.TOK = 0

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

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        tokenizer.pad_token = tokenizer.eos_token
        print(Fore.BLUE + f"[!] Using model: {self.model_name} on device: {device}" + Style.RESET_ALL)
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
                data BLOB,
                tok INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        return conn

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
        if words in self.bloom:
            return
        self.bloom.add(words)

        t0 = time.time()
        for word in words.split():
            try:
                clean_word = self.sanitize(word)
                text, tok = self.llm_generate(clean_word)
                td = time.time() - t0
                yield self.sanitize(text), tok, td, depth
                yield from self.g(text, depth=depth + 1)
            except Exception as e:
                print(Fore.RED + f"[!] Generation error: {e}" + Style.RESET_ALL)

    def distill(self, root_word):
        n = 0
        try:
            with self.conn:
                for text, tok, td, depth in self.g(root_word):
                    self.TOK += tok
                    ts = tok / td if td > 0 else 0
                    print(Fore.YELLOW + f"[+] Generation: {n}, depth: {depth}, tokens: {tok} at {round(ts, 2)} tokens/s\n" +
                          Fore.GREEN + f"[{text}]" + Style.RESET_ALL)
                    data = zlib.compress(text.encode("utf8"))
                    self.conn.execute(
                        "INSERT INTO texts (word, data, tok) VALUES (?, ?, ?)",
                        (root_word, data, tok)
                    )
                    print(f"[+] Total tokens: {self.TOK}")
                    n += 1
        except Exception as e:
            print(Fore.RED + f"[!] DB error: {e}" + Style.RESET_ALL)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_word = sys.argv[1]

        model_name = "distilgpt2"
        #model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        #model_name = "sgugger/rwkv-430M-pile"
        #model_name = "microsoft/bitnet-b1.58-2B-4T"

        distiller = Distiller(model_name="model_name", db_path="words/data.db")
        distiller.distill(input_word)
    else:
        print(Fore.RED + "[!] Please provide an input word." + Style.RESET_ALL)

