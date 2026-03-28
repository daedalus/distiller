import json
import random
import re
import sqlite3
import time
from collections import deque
from urllib.parse import urlparse

from colorama import Fore, Style, init

from ..adapters.bloomfilter import BloomFilter
from ..core.tfidf import TFIDFHelper
from ..core.utils import (
    backup_file,
    compress_content,
    get_json_response,
    get_response,
    parse_arguments,
)

init(autoreset=True)


class Distiller:
    def __init__(
        self,
        model_name: str | None = None,
        db_path: str | None = None,
        max_depth: int = 10,
        compression_level: int = 0,
        seed: int | None = None,
        bloom_size: int = 100000,
        bloom_hash_count: int = 3,
        retrieve_to_bloom: bool = False,
        use_unsloth: bool = False,
        api_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        max_ngrams: int = 6,
        api_hf_provider: bool = False,
        prompt_prefixes: list[str] | None = None,
        batch_size: int = 1,
        remove_prompt: bool = False,
        min_tfidf_score: float = 0.01,
        compression_algo: str = "gz",
        remote_hostname: str = "localhost",
        ngram_mode: bool = False,
        save_to_textfile: bool = False,
        q_mode: bool = False,
        randomize_model_retry: bool = False,
        strip_think_tag_form_prompt: bool = False,
        randomize_remote_endpoint: bool = False,
        secrets: dict | None = None,
        exp_backoff: bool = False,
        stream: bool = False,
        threads: int | None = None,
        load_prompts_from_file: str | None = None,
        randomize_prompts: bool = False,
    ) -> None:
        self.model_name = model_name
        self.db_path = db_path
        self.max_depth = max_depth
        self.compression_level = compression_level
        self.seed = seed
        self.bloom_size = bloom_size
        self.bloom_hash_count = bloom_hash_count
        self.retrieve_to_bloom = retrieve_to_bloom
        self.use_unsloth = use_unsloth
        self.api_url = api_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.max_ngrams = max_ngrams
        self.api_hf_provider = api_hf_provider
        self.prompt_prefixes = prompt_prefixes or []
        self.batch_size = batch_size
        self.remove_prompt = remove_prompt
        self.min_tfidf_score = min_tfidf_score
        self.compression_algo = compression_algo
        self.remote_hostname = remote_hostname
        self.ngram_mode = ngram_mode
        self.save_to_textfile = save_to_textfile
        self.q_mode = q_mode
        self.randomize_model_retry = randomize_model_retry
        self.strip_think_tag_form_prompt = strip_think_tag_form_prompt
        self.randomize_remote_endpoint = randomize_remote_endpoint
        self.secrets = secrets
        self.exp_backoff = exp_backoff
        self.stream = stream
        self.threads = threads
        self.load_prompts_from_file = load_prompts_from_file
        self.randomize_prompts = randomize_prompts

        self.TOK = 0
        self.runtime = 0.0
        self.corpus = []
        self.tfidf_helper = None
        self.bloom = BloomFilter(size=bloom_size, hash_count=bloom_hash_count)

        if db_path:
            self.conn = sqlite3.connect(db_path)
            with self.conn:
                self.conn.execute(
                    "CREATE TABLE IF NOT EXISTS texts (seed, model, word, data, tok, compression_algo, PRIMARY KEY (seed, model, word, data))"
                )
                self.conn.execute(
                    "CREATE TABLE IF NOT EXISTS prompts (id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT UNIQUE, response TEXT, compression_level INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"
                )
        else:
            self.conn = None

        if save_to_textfile:
            self.textfile = open("texts.txt", "a")
        else:
            self.textfile = None

    def generate_batch(self, prompts: list[str]) -> tuple[list[str], list[int]]:
        texts = []
        tokens = []

        for prompt in prompts:
            text, tok = self.generate(prompt)
            texts.append(text)
            tokens.append(tok)

        return texts, tokens

    def generate(self, prompt: str) -> tuple[str, int]:
        if self.strip_think_tag_form_prompt:
            prompt = re.sub(r"<\|.*?\|>", "", prompt, flags=re.DOTALL)

        if self.api_url:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = get_json_response(
                self.api_url,
                self.model_name,
                messages,
                self.max_tokens,
                self.seed,
                self.stream,
                self.exp_backoff,
                self.secrets,
            )

            if isinstance(response, dict):
                text = (
                    response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                tok = response.get("usage", {}).get("total_tokens", 0)
            else:
                text = str(response)
                tok = len(text.split())
        else:
            text, tok = get_response(
                self.api_url,
                self.model_name,
                prompt,
                self.max_tokens,
                self.seed,
                self.stream,
                self.exp_backoff,
                self.secrets,
            )

        return text, tok

    def g(self, root_word: str):
        stack = deque([(root_word, 0)])
        batch = []

        while stack:
            prompt, depth = stack.pop()

            if self.compression_level > 0:
                prompt = compress_content(
                    prompt,
                    self.model_name,
                    self.api_url,
                    self.max_tokens,
                    self.compression_level,
                    self.max_depth,
                    self.prompt_prefixes,
                    self.min_tfidf_score,
                    self.secrets,
                    self.seed,
                    self.exp_backoff,
                )

            batch.append((prompt, depth))

            if len(batch) >= self.batch_size:
                prompts = [item[0] for item in batch]
                depths = [item[1] for item in batch]

                t0 = time.time()
                texts, toks = self.generate_batch(prompts)
                td = time.time() - t0

                for i, (text, tok, depth) in enumerate(zip(texts, toks, depths)):
                    clean_text = text
                    if self.remove_prompt:
                        clean_text = clean_text.replace(prompts[i], "")

                    self.corpus.append(clean_text)

                    if self.ngram_mode:
                        self.tfidf_helper = TFIDFHelper(
                            corpus=self.corpus,
                            min_tfidf_score=self.min_tfidf_score,
                            min_ngrams=1,
                            max_ngrams=self.max_ngrams,
                        )

                    yield clean_text, tok, td, depth

                    new_prompts = []

                    if self.ngram_mode:
                        lines = self.tfidf_helper.all_ngrams(clean_text)
                    else:
                        lines = clean_text.split("\n")

                    for line in lines:
                        for prefix in self.prompt_prefixes:
                            new_prompt = f"{prefix} {line}" if prefix else line

                            if new_prompt in self.bloom:
                                continue
                            self.bloom.add(new_prompt)

                            if depth + 1 < self.max_depth:
                                new_prompts.append((new_prompt, depth + 1))

                    stack.extend(reversed(new_prompts))

                batch = []

        if batch:
            prompts = [item[0] for item in batch]
            depths = [item[1] for item in batch]

            t0 = time.time()
            texts, toks = self.generate_batch(prompts)
            td = time.time() - t0

            for i, (text, tok, depth) in enumerate(zip(texts, toks, depths)):
                clean_text = text
                if self.remove_prompt:
                    clean_text = clean_text.replace(prompts[i], "")

                self.corpus.append(clean_text)

                if self.ngram_mode:
                    self.tfidf_helper = TFIDFHelper(
                        corpus=self.corpus,
                        min_tfidf_score=self.min_tfidf_score,
                        min_ngrams=1,
                        max_ngrams=self.max_ngrams,
                    )

                yield clean_text, tok, td, depth

                new_prompts = []

                if self.ngram_mode:
                    lines = self.tfidf_helper.all_ngrams(clean_text)
                else:
                    lines = clean_text.split("\n")

                for line in lines:
                    for prefix in self.prompt_prefixes:
                        new_prompt = f"{prefix} {line}" if prefix else line

                        if new_prompt in self.bloom:
                            continue
                        self.bloom.add(new_prompt)

                        if depth + 1 < self.max_depth:
                            new_prompts.append((new_prompt, depth + 1))

                stack.extend(reversed(new_prompts))

    def distill(self, root_word: str) -> None:
        time.time()
        self.TOK = 0
        n = 0

        try:
            if self.conn:
                for text, tok, td, depth in self.g(root_word):
                    self.TOK += tok
                    self.runtime += td
                    ts = tok / td if td > 0 else 0
                    print(
                        Fore.YELLOW
                        + f"[+] [{self.remote_hostname}]  Generation: {n}, depth: {depth}, tokens: {tok} at {round(ts, 2)} tokens/s\n"
                        + Fore.GREEN
                        + f"[{text}]"
                        + Style.RESET_ALL
                    )

                    if self.conn:
                        compressed = compress_content(text, self.compression_level)
                        self.conn.execute(
                            "INSERT or ignore INTO texts (seed, model, word, data, tok, compression_algo) VALUES (?,?,?,?,?,?)",
                            (
                                self.seed,
                                self.model_name,
                                root_word,
                                compressed,
                                tok,
                                self.compression_algo,
                            ),
                        )
                        self.conn.commit()

                    if self.save_to_textfile and self.textfile:
                        self.textfile.writelines(text)
                        self.textfile.flush()

                    lt, lc = len(text), len(compressed)
                    print(
                        Fore.BLUE
                        + f"[+] Text size: {lt} bytes, Compressed data size ({self.compression_algo}): {lc} bytes, ratio: {round(lt / lc, 2)}"
                        + Style.RESET_ALL
                    )

                    print(
                        f"[+] Total generated tokens: {self.TOK}, total elapsed time: {round(self.runtime, 2)} seconds, total tokens/s: {round(self.TOK / self.runtime, 2)}"
                    )

                    n += 1
        except Exception as e:
            print(Fore.RED + f"[!] Error: {e}" + Style.RESET_ALL)


def main() -> int:
    args = parse_arguments()

    if args.db:
        backup_file(args.db, "./words/")

    remote_hostname = "localhost"
    if args.api_url is not None:
        remote_hostname = urlparse(args.api_url).hostname
    if args.api_hf_provider:
        remote_hostname = "api.huggingface.co"

    secrets = None
    if args.secrets_file is not None:
        hostname = urlparse(args.api_url).hostname
        secrets = json.load(open(args.secrets_file))
        api_key = secrets[hostname]
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
        use_unsloth=args.use_unsloth,
        api_url=args.api_url,
        api_key=api_key,
        max_tokens=args.max_tokens,
        system_prompt=args.system_prompt,
        max_ngrams=args.max_ngrams,
        api_hf_provider=args.api_hf_provider,
        prompt_prefixes=args.prompt_prefixes,
        batch_size=args.batch_size,
        remove_prompt=args.remove_prompt,
        min_tfidf_score=args.min_tfidf_score,
        compression_algo=args.compression_algo,
        remote_hostname=remote_hostname,
        ngram_mode=args.ngram_mode,
        save_to_textfile=args.save_to_textfile,
        q_mode=args.q_mode,
        randomize_model_retry=args.randomize_model_retry,
        strip_think_tag_form_prompt=args.strip_think_tag_form_prompt,
        randomize_remote_endpoint=args.randomize_remote_endpoint,
        secrets=secrets,
        exp_backoff=args.exp_backoff,
        stream=args.stream,
        threads=args.threads,
        load_prompts_from_file=args.load_prompts_from_file,
        randomize_prompts=args.randomize_prompts,
    )

    if args.load_prompts_from_file:
        prompts = list(open(args.load_prompts_from_file))
        if args.randomize_prompts:
            random.shuffle(prompts)
        for prompt in prompts:
            distiller.distill(prompt)
    elif args.prompt:
        distiller.distill(args.prompt)

    return 0
