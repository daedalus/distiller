import re
import zlib
import zstd
import argparse
from colorama import Fore, Style

def sanitize(s):
    s = re.sub(r'\n+', '\n', s)
    #s = re.sub(r' +', ' ', s)
    return s #.strip()

def compress(text, level=6, algorithm="zlib"):
    binary = text.encode("utf8")
    if algorithm == 'zlib':
       return zlib.compress(binary, level=level)
    if algorithm == 'zstd':
       return zstd.compress(binary, level)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Distiller with Bloom filter and SQLite storage.")
    parser.add_argument("--prompt", default=None, help="Root word or prompt to distill.")
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
    parser.add_argument("--load-prompts-from-file", default=None, help="Specify the prompts file file.")
    parser.add_argument("--api-hf-provider", default=None, help="Specify the hugging face inference provider")    
    parser.add_argument("--compression-algo", default='zlib', help="Specify the compresion algo to use.")    
    parser.add_argument('--prompt-prefixes', nargs='+', help='List of strings with spaces allowed')
    parser.add_argument("--batch-size", type=int, default=1, help="Number of prompts to process in parallel (default: 1)")
    parser.add_argument("--remove-prompt", action="store_true", help="Remove the prompt from the generation.")
    parser.add_argument("--min-tfidf-score",type=float, default=0.2, help="Specify the min_tfidf_score.")


    args = parser.parse_args()

    if args.no_color:
        Fore.RESET = ""
        Style.RESET_ALL = ""
    return args


def all_ngrams(text, n, s):
        text = text.replace("\n", "")
        words = text.split() 
        for size in range(n, s, -1):
            for i in range(n - size + 1):
                yield " ".join(words[i:i + size])


