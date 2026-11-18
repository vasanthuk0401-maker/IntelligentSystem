"""Local sentence embeddings (no API key) via fastembed + ONNX.

Default model: BAAI/bge-small-en-v1.5 (~130 MB, downloaded once into
data/models). Chosen because, on labelled profile/job pairs, it beat TF-IDF
(mean AUC 0.97 vs 0.92) and was much better on CVs that paraphrase skills.

If fastembed or the model is unavailable, `get_embedder()` returns None and
callers fall back to TF-IDF.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

import numpy as np

from core import config

log = logging.getLogger(__name__)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_lock = threading.Lock()
_instance: Optional["Embedder"] = None
_failed = False


class Embedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        from fastembed import TextEmbedding

        self.model_name = model_name
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(model_name, cache_dir=str(config.MODELS_DIR))
        self._cache: dict = {}

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalised embeddings (rows), cached per text."""
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            for t, v in zip(todo, self._model.embed(todo)):
                v = np.asarray(v, dtype=np.float32)
                self._cache[t] = v / (np.linalg.norm(v) or 1.0)
        return np.stack([self._cache[t] for t in texts])


def get_embedder() -> Optional[Embedder]:
    """Lazily load the shared embedder; None if disabled or unavailable."""
    global _instance, _failed
    if config.EMBEDDING_MODEL.lower() in {"", "none", "tfidf"} or _failed:
        return None
    if _instance is None:
        with _lock:
            if _instance is None and not _failed:
                try:
                    _instance = Embedder()
                except Exception as e:  # not installed, offline on first run, etc.
                    log.warning("Embeddings unavailable, using TF-IDF: %s", e)
                    _failed = True
    return _instance
