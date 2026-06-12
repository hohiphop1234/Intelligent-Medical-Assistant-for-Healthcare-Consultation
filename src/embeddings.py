from __future__ import annotations

import math
import re
from typing import Any

from config import (
    EMBEDDING_MODEL_EN,
    EMBEDDING_MODEL_VI,
    FALLBACK_EMBEDDING_DIM,
    FORCE_FALLBACK_EMBEDDINGS,
)
from src.utils import normalize_for_match, stable_hash, tokenize

try:
    from langdetect import detect
except ImportError:  # pragma: no cover - optional dependency
    detect = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None


VI_HINTS = {
    "thuoc",
    "benh",
    "trieu",
    "dieu",
    "tac",
    "dung",
    "phu",
    "cua",
    "la",
    "gi",
    "co",
    "khong",
    "thai",
    "tre",
    "nguoi",
    "dau",
    "sot",
    "lieu",
    "tuong",
}


class DualEmbeddingManager:
    """Language-aware embedding manager with a deterministic fallback."""

    def __init__(self, allow_fallback: bool = True):
        self.allow_fallback = allow_fallback
        self._models: dict[str, Any] = {}
        self._cache: dict[str, list[float]] = {}
        self._model_failed: set[str] = set()

    def detect_language(self, text: str) -> str:
        normalized = normalize_for_match(text)
        if re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", text.lower()):
            return "vi"
        if any(hint in normalized.split() for hint in VI_HINTS):
            return "vi"
        if detect is not None:
            try:
                return "vi" if detect(text) == "vi" else "en"
            except Exception:
                pass
        return "en"

    def embed(self, text: str, language: str | None = None) -> list[float]:
        if language is None:
            language = self.detect_language(text)
        cache_key = stable_hash(f"{language}:{text}", 32)
        if cache_key in self._cache:
            return self._cache[cache_key]
        embedding = self.embed_batch([text], language=language)[0]
        self._cache[cache_key] = embedding
        return embedding

    def embed_batch(self, texts: list[str], language: str) -> list[list[float]]:
        model = self._load_model(language)
        if model is not None:
            try:
                return model.encode(texts, show_progress_bar=len(texts) > 8).tolist()
            except Exception:
                self._model_failed.add(language)
                if not self.allow_fallback:
                    raise
        return [self._fallback_embedding(text) for text in texts]

    def get_embedding_dim(self, language: str) -> int:
        model = self._load_model(language)
        if model is not None:
            try:
                return int(model.get_sentence_embedding_dimension())
            except Exception:
                pass
        return FALLBACK_EMBEDDING_DIM

    def _load_model(self, language: str) -> Any | None:
        if FORCE_FALLBACK_EMBEDDINGS or SentenceTransformer is None or language in self._model_failed:
            return None
        if language in self._models:
            return self._models[language]
        model_name = EMBEDDING_MODEL_VI if language == "vi" else EMBEDDING_MODEL_EN
        try:
            self._models[language] = SentenceTransformer(model_name)
            return self._models[language]
        except Exception:
            self._model_failed.add(language)
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
