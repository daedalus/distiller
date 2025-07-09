## The distiller ##

### Running ###

```
usage: main.py [-h] [--model MODEL] [--db DB] [--max-depth MAX_DEPTH] [--max-tokens MAX_TOKENS] [--compression-level {1,2,3,4,5,6,7,8,9}] [--seed SEED] [--bloom-size BLOOM_SIZE]
               [--bloom-hash-count BLOOM_HASH_COUNT] [--max-ngrams MAX_NGRAMS] [--no-color] [--retrieve-to-bloom] [--use-unsloth] [--api-url API_URL] [--api-key API_KEY]
               [--system-prompt SYSTEM_PROMPT] [--threads THREADS] [--secrets-file SECRETS_FILE] [--api-hf-provider API_HF_PROVIDER]
               [--prompt-prefixes PROMPT_PREFIXES [PROMPT_PREFIXES ...]] [--batch-size BATCH_SIZE] [--remove-prompt] [--min-tfidf-score MIN_TFIDF_SCORE]
               prompt
main.py: error: the following arguments are required: prompt
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
