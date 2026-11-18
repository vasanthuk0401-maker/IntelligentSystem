"""Central configuration, loaded from environment / .env."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = Path(os.getenv("MEMORY_DB_PATH", DATA_DIR / "memory.db"))
JOBS_DATASET = Path(os.getenv("JOBS_DATASET", DATA_DIR / "jobs.json"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", DATA_DIR / "models"))
# Local embedding model for semantic matching; set to "tfidf" to disable.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "60"))
MAX_SEARCH_ITERATIONS = int(os.getenv("MAX_SEARCH_ITERATIONS", "3"))
TOP_K_ANALYZE = int(os.getenv("TOP_K_ANALYZE", "10"))

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "")
ADZUNA_COUNTRY = os.getenv("ADZUNA_COUNTRY", "in")
