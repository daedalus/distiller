"""Configuration management for llm_distiller.

This module provides functionality to load and manage configuration from YAML files.
"""

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Configuration manager that loads settings from a YAML file.

    Attributes:
        path: Path to the configuration file.
        data: Dictionary containing configuration values.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        """Initialize the configuration.

        Args:
            path: Path to the YAML configuration file. If None, uses default config.
        """
        self.path = Path(path) if path else Path("config.yaml")
        self.data: dict[str, Any] = {}
        if self.path.exists():
            self.load()

    def load(self) -> None:
        """Load configuration from the YAML file."""
        with open(self.path, encoding="utf-8") as f:
            self.data = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key (supports dot notation for nested keys).
            default: Default value if key is not found.

        Returns:
            The configuration value or default.
        """
        keys = key.split(".")
        value = self.data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key: Configuration key (supports dot notation for nested keys).
            value: Value to set.
        """
        keys = key.split(".")
        data = self.data
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]
        data[keys[-1]] = value

    def save(self) -> None:
        """Save configuration to the YAML file."""
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.data, f, default_flow_style=False, sort_keys=False)

    def __getitem__(self, key: str) -> Any:
        """Get a configuration value using bracket notation."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a configuration value using bracket notation."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if a configuration key exists."""
        return self.get(key) is not None


DEFAULT_CONFIG = """# Distiller Configuration
# Copy this to config.yaml and customize as needed

model: distilgpt2
api_url: null
api_key: null

# Generation settings
max_tokens: 1024
max_depth: 10
compression_level: 0

# Database
db: words/data.db

# TF-IDF settings
min_tfidf_score: 0.01
min_ngrams: 1
max_ngrams: 6

# Bloom filter
bloom_filter_size: 27
bloom_hash_count: 8

# Output
no_color: false
verbose: false

# Retry settings
exp_backoff: false
"""


def create_default_config(path: str | Path | None = None) -> Config:
    """Create a configuration file with default values.

    Args:
        path: Path where to create the config file. Defaults to config.yaml.

    Returns:
        Config instance with default values.
    """
    config_path = Path(path) if path else Path("config.yaml")
    if not config_path.exists():
        config_path.write_text(DEFAULT_CONFIG, encoding="utf-8")
    return Config(config_path)
