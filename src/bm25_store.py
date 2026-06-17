from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

from src.utils import tokenize

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - optional dependency
    BM25Okapi = None


class BM25Store:
    """Keyword index used beside vector search."""

    def __init__(self):
        self.indices: dict[str, Any] = {"vi": None, "en": None}
        self.documents: dict[str, list[str]] = {"vi": [], "en": []}
        self.doc_ids: dict[str, list[str]] = {"vi": [], "en": []}
        self.metadatas: dict[str, list[dict[str, Any]]] = {"vi": [], "en": []}
        self.tokenized: dict[str, list[list[str]]] = {"vi": [], "en": []}
        self._idf: dict[str, dict[str, float]] = {"vi": {}, "en": {}}

    def build_index(self, chunks: list[dict[str, Any]], language: str) -> None:
        docs = [self._tokenize(chunk["content"], language) for chunk in chunks]
        self.tokenized[language] = docs
        self.documents[language] = [chunk["content"] for chunk in chunks]
        self.doc_ids[language] = [str(chunk["id"]) for chunk in chunks]
        self.metadatas[language] = [self._metadata(chunk) for chunk in chunks]
        if BM25Okapi is not None:
            self.indices[language] = BM25Okapi(docs)
        else:
            self.indices[language] = "simple"
            self._idf[language] = self._build_simple_idf(docs)

    def search(
        self,
        query: str,
        language: str,
        top_k: int = 5,
        category_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        if self.indices[language] is None:
            return []
        query_tokens = self._tokenize(query, language)
        if not query_tokens:
            return []

        if BM25Okapi is not None and self.indices[language] != "simple":
            raw_scores = self.indices[language].get_scores(query_tokens)
            scores = [float(score) for score in raw_scores]
        else:
            scores = self._simple_scores(query_tokens, language)

        ranked_indices = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )
        results = []
        for index in ranked_indices:
            if len(results) >= top_k:
                break
            if scores[index] <= 0:
                continue
            metadata = self.metadatas[language][index]
            if category_filter and metadata.get("category") != category_filter:
                continue
            results.append(
                {
                    "id": self.doc_ids[language][index],
                    "doc_id": self.doc_ids[language][index],
                    "content": self.documents[language][index],
                    "score": float(scores[index]),
                    "metadata": metadata,
                }
            )
        return results

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "indices": self.indices,
                    "documents": self.documents,
                    "doc_ids": self.doc_ids,
                    "metadatas": self.metadatas,
                    "tokenized": self.tokenized,
                    "idf": self._idf,
                },
                f,
            )

    def load(self, path: str) -> bool:
        if not Path(path).exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.indices = data["indices"]
        self.documents = data["documents"]
        self.doc_ids = data["doc_ids"]
        self.metadatas = data.get("metadatas", {"vi": [], "en": []})
        self.tokenized = data.get("tokenized", {"vi": [], "en": []})
        self._idf = data.get("idf", {"vi": {}, "en": {}})
        return True

    def _tokenize(self, text: str, language: str) -> list[str]:
        return tokenize(text)

    def _metadata(self, chunk: dict[str, Any]) -> dict[str, Any]:
        return {
            "source": chunk.get("source", ""),
            "entity": chunk.get("entity", ""),
            "category": chunk.get("category", chunk.get("topic_group", "")),
            "topic_group": chunk.get("topic_group", ""),
            "risk_level": chunk.get("risk_level", "medium"),
            "section": chunk.get("section", ""),
            "url": chunk.get("url", ""),
            "title": chunk.get("title", ""),
            "language": chunk.get("language", ""),
        }

    def _build_simple_idf(self, docs: list[list[str]]) -> dict[str, float]:
        doc_count = max(len(docs), 1)
        doc_freq: dict[str, int] = {}
        for doc in docs:
            for token in set(doc):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        return {
            token: math.log((doc_count - freq + 0.5) / (freq + 0.5) + 1)
            for token, freq in doc_freq.items()
        }

    def _simple_scores(self, query_tokens: list[str], language: str) -> list[float]:
        idf = self._idf[language]
        scores = []
        for doc in self.tokenized[language]:
            doc_length = max(len(doc), 1)
            term_counts: dict[str, int] = {}
            for token in doc:
                term_counts[token] = term_counts.get(token, 0) + 1
            score = 0.0
            for token in query_tokens:
                tf = term_counts.get(token, 0)
                if tf == 0:
                    continue
                score += idf.get(token, 0.0) * (tf / doc_length) * 100
            scores.append(score)
        return scores
