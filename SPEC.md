# SPEC.md — llm-distiller

## Purpose
Distiller is a CLI tool that extracts key information from LLM (Large Language Model) outputs using TF-IDF (Term Frequency-Inverse Document Frequency) analysis. It processes text prompts and LLM responses to identify the most relevant n-grams based on their TF-IDF scores.

## Scope

### In Scope
- CLI interface for running LLM inference with compression
- TF-IDF-based text analysis for extracting key information
- Support for both local models (via Unsloth) and remote API endpoints
- Configurable n-gram extraction (min/max n-grams)
- SQLite database storage for tracking prompts and responses
- Bloom filter for deduplication
- Batch processing support
- Seed-based reproducibility

### Not in Scope
- GUI interface
- Web server/API
- Model training or fine-tuning
- Multi-language support beyond English

## Public API / Interface

### CLI Commands

```bash
llm-distiller <prompt> [OPTIONS]
```

**Options:**
- `--api-url TEXT` - API URL for remote inference
- `--model TEXT` - Model name to use
- `--max-tokens INTEGER` - Maximum tokens to generate (default: 1024)
- `--max-depth INTEGER` - Maximum recursion depth (default: 10)
- `--compression-level INTEGER` - Compression level 0-9 (default: 0)
- `--batch-size INTEGER` - Batch size for processing (default: 1)
- `--seed INTEGER` - Random seed for reproducibility
- `--db TEXT` - Path to SQLite database
- `--prompt-prefixes TEXT` - Prefixes to add to prompts (multiple)
- `--remove-prompt` - Remove prompt from output
- `--min-tfidf-score FLOAT` - Minimum TF-IDF score threshold (default: 0.01)
- `--use-unsloth` - Use Unsloth for faster inference
- `--no-color` - Disable colored output
- `--exp-backoff` - Enable exponential backoff
- `--stream` - Enable streaming responses

### Python API

#### TFIDFHelper Class
```python
class TFIDFHelper:
    def __init__(self, corpus: list[str] = [], min_tfidf_score: float = 0.01, 
                 min_ngrams: int = 1, max_ngrams: int = 6)
    def get_tfidf_scores(self, text: str) -> dict[str, float]
    def all_ngrams(self, text: str) -> Iterator[str]
```

#### Utils Functions
```python
def get_response(api_url: str, model: str, prompt: str, max_tokens: int, 
                 seed: int | None, stream: bool, exp_backoff: bool, 
                 secrets: dict) -> str
def get_json_response(api_url: str, model: str, messages: list[dict], 
                      max_tokens: int, seed: int | None, stream: bool, 
                      exp_backoff: bool, secrets: dict) -> dict
def compress_content(content: str, ngrams: int) -> str
def expand_content(content: str, model: str, api_url: str, max_tokens: int, 
                   compression_level: int, max_depth: int, 
                   prompt_prefixes: list[str], min_tfidf_score: float,
                   secrets: dict, seed: int | None, exp_backoff: bool) -> str
def parse_arguments() -> argparse.Namespace
def all_ngrams(text: str, n: int, s: int) -> Iterator[str]
def backup_file(filepath: str, path: str) -> str
```

## Data Formats

### Input
- Text prompts (strings)
- JSON secrets file for API keys
- SQLite database for persistence

### Output
- Compressed/expanded text content
- TF-IDF scores as dictionaries
- N-grams as strings

### Database Schema
```sql
CREATE TABLE IF NOT EXISTS prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT NOT NULL,
    response TEXT,
    compression_level INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## Edge Cases

1. **Empty corpus**: TF-IDF vectorizer requires non-empty corpus - handle gracefully
2. **API timeout**: Implement exponential backoff for failed requests
3. **Large text inputs**: Limit n-gram range to prevent memory issues
4. **Invalid JSON responses**: Handle malformed API responses gracefully
5. **Database lock errors**: Handle SQLite concurrent access
6. **Missing API keys**: Raise clear error messages
7. **Model not found**: Handle missing/invalid model names
8. **Rate limiting**: Implement backoff for API rate limits
9. **Empty prompts**: Handle edge case of empty or whitespace-only prompts
10. **Seed reproducibility**: Ensure same seed produces identical outputs

## Performance & Constraints

- TF-IDF vectorization: O(corpus_size * avg_doc_length)
- N-gram generation: O(text_length^2) in worst case, bounded by max_ngrams
- Database operations: O(log n) for inserts/queries
- Memory: Limit max_ngrams to prevent excessive memory consumption
- No external dependencies in core TFIDFHelper (sklearn is used for TF-IDF)
