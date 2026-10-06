"""Defaults shared by the experiment commands."""

from pathlib import Path

DEFAULT_DATASET = "irds:lotte/technology/dev/search"
DEFAULT_MAX_DOCS = 5000
DEFAULT_INDEX = Path("data/index_lotte_5000")
DEFAULT_ENCODER = "distilbert-base-uncased"
DEFAULT_GENERATOR = "Qwen/Qwen3-0.6B"
