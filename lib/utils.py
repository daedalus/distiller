import os
import re
import zlib
import zstd
import time
import argparse
import subprocess
from colorama import Fore, Style
from pathlib import Path
from shutil import copy2


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


def decompress(text, algorithm="zlib"):
    if algorithm == 'zlib': r = zlib.decompress(text)
    if algorithm == 'zstd': r = zstd.decompress(text)
    return r.decode("utf8")   


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
    parser.add_argument("--remove-prompt", action="store_true", default=False, help="Remove the prompt from generation.")
    parser.add_argument("--ngram-mode", action="store_true", default=False, help="ngram mode from generation.")
    parser.add_argument("--min-tfidf-score",type=float, default=0.2, help="Specify the min_tfidf_score.")
    parser.add_argument("--save-to-textfile", default=None, help="Specify a text file to save generated text.")
    parser.add_argument("--q-mode", action="store_true", default=None, help="Q-mode.")
    parser.add_argument("--randomize-prompts", action="store_true", default=None, help="Randomize prompts when read from file.")
    parser.add_argument("--randomize-model-retry", action="store_true", default=None, help="Randomize model to retry.")
    parser.add_argument("--randomize-remote-endpoint", action="store_true", default=None, help="Randomize remote endpoint.")
    parser.add_argument("--strip-think-tag-form-prompt", action="store_true", default=None, help="Strip the think tag from prompts.")
    parser.add_argument("--exp-backoff", action="store_true", default=False, help="Set exponential backoff.")


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


def backup_file(filepath, path):
    """
    Create a backup of the specified file in the given directory.

    The backup file will be named:
        .$name.$timestamp.$extension.bkp

    A reflink copy is attempted first (if supported), falling back to a full copy.
    Keeps only the 3 most recent backups, deleting older ones.

    Args:
        filepath (str): Path to the source file.
        path (str): Destination directory for the backup.

    Returns:
        str: Path to the created backup file.

    Raises:
        FileNotFoundError: If the source file does not exist.
        Exception: If both reflink and standard copy fail.
    """
    src = Path(filepath)
    dst_dir = Path(path)

    if not src.is_file():
        raise FileNotFoundError(f"Source file does not exist: {filepath}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    name = src.stem
    ext = src.suffix.lstrip(".")
    timestamp = int(time.time())
    backup_name = f".{name}.{timestamp}.{ext}.bkp"
    dst = dst_dir / backup_name

    try:
        subprocess.run(["cp", "--reflink=auto", str(src), str(dst)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            copy2(src, dst)
        except Exception as e:
            raise Exception(f"Failed to back up file: {e}")

    # Delete older backups, keeping only the 3 most recent
    pattern = f".{name}.*.{ext}.bkp"
    backups = sorted(dst_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    for old in backups[3:]:
        try:
            old.unlink()
        except Exception as e:
            print(f"Warning: failed to delete old backup {old}: {e}")

    return str(dst)

