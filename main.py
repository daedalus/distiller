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
import sqlite3
import torch
import time
import argparse
import openai
import json
import random
from typing import List, Tuple, Generator
from transformers import AutoTokenizer, AutoModelForCausalLM
from urllib.parse import urlparse
from huggingface_hub import InferenceClient
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Generator, Dict
from joblib import Memory
from lib.bloomfilter import BloomFilter
from lib.utils import sanitize, compress, decompress, parse_args, backup_file
from lib.TFIDFHelper import TFIDFHelper


class Distiller:
    def __init__(self, model_name, db_path, max_depth=10, compression_level=6, seed=None, 
                 bloom_size=24, bloom_hash_count=6, retrieve_to_bloom=False, use_unsloth=False, 
                 max_tokens=1024, api_url=None, api_key=None, system_prompt=None, 
                 max_ngrams=10, min_ngrams=2, api_hf_provider=None, prompt_prefixes=None,
                 batch_size=1, min_tfidf_score=0.2, tfidf_cache_dir=None, remove_prompt = False, 
                 compression_algo='zlib', remote_hostname='localhost', ngram_mode=False, save_to_textfile=None, 
                 q_mode=False, randomize_model_retry=False, strip_think_tag_form_prompt = False,
                 randomize_remote_endpoint = False, secrets=None, exp_backoff=False, stream=False ):

        # Initialize core parameters
        self.model_name = model_name
        self.db_path = db_path
        self.max_depth = max_depth
        self.compression_level = compression_level
        self.bloom = BloomFilter(power2=bloom_size, hash_count=bloom_hash_count)
        self.TOK = 0
        self.runtime = 0
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
        self.api_hf_provider = api_hf_provider
        self.prompt_prefixes = prompt_prefixes if prompt_prefixes is not None else ['']
        self.remove_prompt = remove_prompt
        self.compression_algo = compression_algo     
        self.remote_hostname = remote_hostname
        self.ngram_mode = ngram_mode 
        self.save_to_textfile = save_to_textfile
        self.q_mode = q_mode
        self.randomize_model_retry = randomize_model_retry
        self.strip_think_tag_from_prompt = strip_think_tag_form_prompt 
        self.randomize_remote_endpoint = randomize_remote_endpoint 
        self.secrets = secrets
        self.exp_backoff = int(exp_backoff)
        self.exp_backoff_endpoint = {}

        self.stream = stream 

        if stream:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.corpus = []
    
 
        if self.randomize_model_retry or self.randomize_remote_endpoint:
            self.inventory = json.load(open('inventory.json','r'))['inventory'] #[self.remote_hostname]

        """
        if self.q_mode:
           with open(".q_done.txt","r") as fp:
               for line in fp:
                   print("PRELOADING:",line)
                   self.bloom.add(line.strip())
        """
        if self.save_to_textfile:
            print(Fore.BLUE + f"[!] Saving to textfile: {self.save_to_textfile}" + Style.RESET_ALL)
            self.textfile = open(self.save_to_textfile, "a")

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

        print(Fore.BLUE + f"[-] Using {self.compression_algo}  compression algorithm." + Style.RESET_ALL)

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
                compression_algo TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
        conn.execute("""
            CREATE TABLE If NOT EXISTS prompts (
                id INTEGER PRIMARY KEY,
                prompt TEXT NOT NULL UNIQUE
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
            cursor.execute("SELECT data,compression_algo FROM texts")
            rows = cursor.fetchall()
            for row in rows:
                self.bloom.add(decompress(*row))        
            print(Fore.GREEN + "[+] Retrieved words from database to Bloom filter." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"[!] Error retrieving words from database: {e}" + Style.RESET_ALL)

        if self.q_mode:
            try:
                cursor = self.conn.cursor()
                cursor.execute("select prompt from prompts;")
                for prompt in cursor.fetchall():
                    self.bloom.add(prompt[0].strip())
                print(Fore.GREEN + "[+] Retrieved prompts from database to Bloom filter." + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f"[!] Error retrieving prompts from database: {e}" + Style.RESET_ALL)

    def llm_local_generate(self, prompt):
        print(Fore.YELLOW + f"[+] localost Processing prompt: {prompt}" + Style.RESET_ALL)
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
        print(Fore.YELLOW + f"[+] Processing prompt: {str(prompts)}" + Style.RESET_ALL)
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


    def backoff_wait(self):
        if self.api_url in self.exp_backoff_endpoint:
            self.exp_backoff_endpoint[self.api_url] <<= self.exp_backoff
        else:
            self.exp_backoff_endpoint[self.api_url] = 1

        self.retry_sleep = self.exp_backoff_endpoint[self.api_url]
        time.sleep(self.retry_sleep)


    def switch_endpoint_and_model(self):
        if self.randomize_remote_endpoint:
            endpoints = self.inventory['remote_endpoints_api']
            hostname = random.choice([e for e in endpoints])
            self.api_url = endpoints[hostname]['url']
            self.remote_hostname = hostname
            self.api_key = self.secrets[hostname]
            self.stream |= endpoints[hostname]['stream']
 
            if self.stream:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

            self.api_client = openai.OpenAI(base_url=self.api_url, api_key=self.api_key)
            print(f"[!] [{self.remote_hostname}] Switched to new provider {self.api_url}.")

        if self.randomize_model_retry:
            print("randomize model")
            self.api_model = random.choice(self.inventory['remote_endpoints_models'][self.remote_hostname])
            self.model_name = self.api_model
            print(f"[!] [{self.remote_hostname}] Switched to new model {self.api_model}.")
 

    def llm_api_generate(self, prompt):
        print(Fore.YELLOW + f"[+] [{self.remote_hostname}] Processing prompt: {prompt}, stream:{self.stream}" + Style.RESET_ALL)
        try:
            response = self.api_client.chat.completions.create(
                model=self.api_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                stream = self.stream,
            )
            self.retry_sleep = 1
            """
            try:
                return response.choices[0].message.content, response.usage.completion_tokens
            except Exception as e:
                print(e)
                print(response, type(response), dir(response))
            """     
            if isinstance(response, openai.Stream):
                data = ""
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        data += chunk.choices[0].delta.content
                tok = len(self.tokenizer.encode(data))
                return data, tok
            return response.choices[0].message.content, response.usage.completion_tokens       

                 
        except Exception as e:
            print(Fore.RED +  f"[!] [{self.remote_hostname}] Error generating commit message: {e} retry in {self.retry_sleep}" + Style.RESET_ALL)
            self.backoff_wait()
            self.switch_endpoint_and_model()
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


    def g(self, root_prompt) -> Generator[Tuple[str, int, float, int], None, None]:


        print(f"{self.ngram_mode}")
        stack = [(root_prompt, 0)]
    
        while stack:
            batch = []
            while len(batch) < self.batch_size and stack:
                prompt, depth = stack.pop()
 
                if self.strip_think_tag_from_prompt:
                    prompt = cleaned = re.sub(r'<think>.*?</think>', '', prompt, flags=re.DOTALL)

                """
                if prompt.strip() not in self.bloom:
                    batch.append((prompt, depth))
                    self.bloom.add(prompt.strip())        
                else:
                    print("in bloom", prompt)
                """
                batch.append((prompt,depth))

            if not batch:
                continue
                
            prompts = [item[0] for item in batch]
            depths = [item[1] for item in batch]
            
            t0 = time.time()
            texts, toks = self.generate_batch(prompts)
            td = time.time() - t0
            
            for i,(text, tok, depth) in enumerate(zip(texts, toks, depths)):
                clean_text = sanitize(text)
                if self.remove_prompt:
                    clean_text = clean_text.replace(prompts[i], '')

                # Initialize TF-IDF components
                self.corpus.append(clean_text)

                if self.ngram_mode:
                    self.tfidf_helper = TFIDFHelper(corpus = self.corpus, min_tfidf_score = self.min_tfidf_score, min_ngrams = self.min_ngrams, max_ngrams = self.max_ngrams)
         
                yield clean_text, tok, td, depth
                
                new_prompts = []
        
                if self.ngram_mode:
                    lines = self.tfidf_helper.all_ngrams(clean_text) 
                else: 
                    lines = clean_text.split("\n")
                    print(f"Lines split: {len(lines)}")

                for line in lines:
                    for prefix in self.prompt_prefixes:
                        new_prompt = f"{prefix} {line}" if prefix else line
                   
                        if new_prompt in self.bloom:
                            continue
                        self.bloom.add(new_prompt)

                        if depth + 1 < self.max_depth:
                            new_prompts.append((new_prompt, depth + 1))
            
                stack.extend(reversed(new_prompts))


    def distill(self, root_word):
        t0 = time.time()
        self.TOK = 0
        n = 0
        if self.q_mode:
            self.q_file = open(".q_done.txt","a")
            with self.q_file as fp_qmode:
                fp_qmode.write(root_word)
                fp_qmode.flush()
        try:
            with self.conn:
                for text, tok, td, depth in self.g(root_word):
                    self.TOK += tok
                    self.runtime += td
                    ts = tok / td if td > 0 else 0
                    print(Fore.YELLOW + f"[+] [{self.remote_hostname}]  Generation: {n}, depth: {depth}, tokens: {tok} at {round(ts, 2)} tokens/s\n" +
                          Fore.GREEN + f"[{text}]" + Style.RESET_ALL)
                    data = compress(text, self.compression_level, self.compression_algo)
                    self.conn.execute(
                        "INSERT or ignore INTO texts (seed, model, word, data, tok, compression_algo) VALUES (?,?,?,?,?,?)",
                        (self.seed, self.model_name, root_word, data, tok, self.compression_algo)
                    )
                    if self.q_mode:
                        #self.conn.execute(f"INSERT or ignore INTO prompts (prompt) values ('{root_word}');")                    
                        self.conn.execute(f"INSERT or ignore INTO prompts (prompt) values (?)" , (root_word.strip(),))                    

                    self.conn.commit()

                    if self.save_to_textfile:
                        with self.textfile as fp_textfile:
                          fp_textfile.writelines(text)
                          fp_textfile.textfile.flush()

                    lt, lc = len(text), len(data)
                    print(Fore.BLUE + f"[+] Text size: {lt} bytes, Compressed data size ({self.compression_algo}): {lc} bytes, ratio: {round(lt/lc,2)}" + Style.RESET_ALL)
                    
                    print(f"[+] Total generated tokens: {self.TOK}, total elapsed time: {round(self.runtime, 2)} seconds, total tokens/s: {round(self.TOK/self.runtime, 2)}")

                    if n % 10 == 0:                  
                        cursor = self.conn.cursor()
                        cursor.execute("select sum(tok) from texts")
                        tokdb = cursor.fetchone()[0]
                        print(f"[+] Total accumulated tokens in database: {round(tokdb/1000000,2)}M.")
                    n += 1
        except Exception as e:
            print(Fore.RED + f"[!] DB error: {e}" + Style.RESET_ALL)

if __name__ == '__main__':
    args = parse_args()

    if args.db:
       backup_file(args.db, "./words/")

    hostname = 'localhost'
    if args.api_url is not None:
        remote_hostname = urlparse(args.api_url).hostname
    if args.api_hf_provider:
        remote_hostname = 'api.hugginface.co' 
    

    if args.threads is not None:
        torch.set_num_threads(args.threads)
        print(Fore.BLUE + f"[-] Torch set to use {args.threads} CPU threads" + Style.RESET_ALL)

    if args.secrets_file is not None:
        hostname = urlparse(args.api_url).hostname

        secrets = json.load(open(args.secrets_file,"r"))
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
        use_unsloth=args.use_unsloth and UNSLOTH_AVAILABLE,
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
        compression_algo = args.compression_algo,
        remote_hostname = remote_hostname,
        ngram_mode = args.ngram_mode,
        save_to_textfile = args.save_to_textfile,
        q_mode = args.q_mode,
        randomize_model_retry = args.randomize_model_retry,
        strip_think_tag_form_prompt = args.strip_think_tag_form_prompt,
        randomize_remote_endpoint = args.randomize_remote_endpoint,
        secrets = secrets,
        exp_backoff = args.exp_backoff,
        stream = args.stream
    )
    if args.load_prompts_from_file:
        prompts = [prompt for prompt in open(args.load_prompts_from_file)]
        if args.randomize_prompts:
            random.shuffle(prompts)
        for prompt in prompts:
            distiller.distill(prompt) 
    if args.prompt:
        distiller.distill(args.prompt)
