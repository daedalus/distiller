## The distiller ##

### Running ###

```
usage: main.py [-h] [--model MODEL] [--db DB] [--max-depth MAX_DEPTH] [--max-tokens MAX_TOKENS] [--compression-level {1,2,3,4,5,6,7,8,9}] [--seed SEED] [--bloom-size BLOOM_SIZE]
               [--bloom-hash-count BLOOM_HASH_COUNT] [--max-ngrams MAX_NGRAMS] [--no-color] [--retrieve-to-bloom] [--use-unsloth] [--api-url API_URL] [--api-key API_KEY]
               [--system-prompt SYSTEM_PROMPT] [--threads THREADS] [--secrets-file SECRETS_FILE] [--api-hf-provider API_HF_PROVIDER]
               [--prompt-prefixes PROMPT_PREFIXES [PROMPT_PREFIXES ...]] [--batch-size BATCH_SIZE] [--remove-prompt] [--min-tfidf-score MIN_TFIDF_SCORE]
               prompt

LLM Distiller with Bloom filter and SQLite storage.

positional arguments:
  prompt                Root word or prompt to distill.

options:
  -h, --help            show this help message and exit
  --model MODEL         Huggingface model name (default: distilgpt2).
  --db DB               Path to SQLite database (default: words/data.db).
  --max-depth MAX_DEPTH
                        Max recursion depth (default: 10).
  --max-tokens MAX_TOKENS
                        Max tokens (default: 1024).
  --compression-level {1,2,3,4,5,6,7,8,9}
                        Zlib compression level (1-9, default: 6).
  --seed SEED           Torch manual seed (optional).
  --bloom-size BLOOM_SIZE
                        Bloom filter size (default: 100,000,000).
  --bloom-hash-count BLOOM_HASH_COUNT
                        Bloom filter hash count (default: 6).
  --max-ngrams MAX_NGRAMS
                        Max ngrams (default: 10).
  --no-color            Disable colored output.
  --retrieve-to-bloom   Retrieve words from the database to the Bloom filter.
  --use-unsloth         Use unsloth
  --api-url API_URL     OpenAI compatible API url.
  --api-key API_KEY     API key for auth.
  --system-prompt SYSTEM_PROMPT
                        System prompt
  --threads THREADS     Number of CPU threads for PyTorch (default: auto)
  --secrets-file SECRETS_FILE
                        Specify the secrets json file.
  --api-hf-provider API_HF_PROVIDER
                        Specify the hugging face inference provider
  --prompt-prefixes PROMPT_PREFIXES [PROMPT_PREFIXES ...]
                        List of strings with spaces allowed
  --batch-size BATCH_SIZE
                        Number of prompts to process in parallel (default: 1)
  --remove-prompt       Remove the prompt from the generation.
  --min-tfidf-score MIN_TFIDF_SCORE
                        Specify the min_tfidf_score.

```


With a local endpoint:

```
#!/bin/bash
set -x

PROMPT='make a list of the most important people in history'
MODEL=meta-llama/llama-4-scout-17b-16e-instruct

python main.py "$PROMPT" \
     --compression-level 9 \
     --max-tokens=2048 \
     --max-depth 100 \
     --seed=0 \
     --model=$MODEL \
     --use-unsloth \
     --db /content/drive/MyDrive/IA/data.db \
     --prompt-prefixes 'please explain' 'please elaborate' 'think about' 'formulate a theory about' 'demonstrate that' \
     --batch-size 8 \
     --remove-prompt \
     --min-tfidf-score=0.1
```

With a remote inference endpoint:
```
#!/bin/bash
set -x

PROMPT='make a list of the most important people in history'
PROVIDER=https://api.groq.com/openai/v1/
MODEL=meta-llama/llama-4-scout-17b-16e-instruct

python main.py "$PROMPT" \
  --api-url=$PROVIDER \
  --model=$MODEL \
  --max-tokens=4096 \
  --prompt-prefixes "please explain" "please elaborate" "think about" "formulate a theory about" "demonstrate that" \
  --remove-prompt \
  --secrets-file=.secrets.json \
  --min-tfidf-score=0.1
```

The license is MIT.
