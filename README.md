## The distiller ##

### Running ###

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
