from __future__ import annotations

import math
from typing import Any

from config import (
    EMBEDDING_MODEL_VI,
    FALLBACK_EMBEDDING_DIM,
    FORCE_FALLBACK_EMBEDDINGS,
)
from src.utils import tokenize, stable_hash

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

class EmbeddingManager:
    """Embedding manager with a deterministic fallback."""

    def __init__(self, allow_fallback: bool = True):
        self.allow_fallback = allow_fallback
        self._model: Any | None = None
        self._cache: dict[str, list[float]] = {}
        self._model_failed: bool = False

    def embed(self, text: str) -> list[float]:
        cache_key = stable_hash(text, 32)
        if cache_key in self._cache:
            return self._cache[cache_key]
        embedding = self.embed_batch([text])[0]
        self._cache[cache_key] = embedding
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        if model is not None:
            try:
                return model.encode(texts, show_progress_bar=len(texts) > 8).tolist()
            except Exception:
                self._model_failed = True
                if not self.allow_fallback:
                    raise
        return [self._fallback_embedding(text) for text in texts]

    def get_embedding_dim(self) -> int:
        model = self._load_model()
        if model is not None:
            try:
                return int(model.get_sentence_embedding_dimension())
            except Exception:
                pass
        return FALLBACK_EMBEDDING_DIM

    def _load_model(self) -> Any | None:
        if FORCE_FALLBACK_EMBEDDINGS or SentenceTransformer is None or self._model_failed:
            return None
        if self._model is not None:
            return self._model
        try:
            self._model = SentenceTransformer(EMBEDDING_MODEL_VI)
            return self._model
        except Exception:
            self._model_failed = True
            if not self.allow_fallback:
                raise
            return None

    def _fallback_embedding(self, text: str) -> list[float]:
        vector = [0.0] * FALLBACK_EMBEDDING_DIM
        tokens = tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = stable_hash(token, 16)
            bucket = int(digest[:8], 16) % FALLBACK_EMBEDDING_DIM
            sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]
