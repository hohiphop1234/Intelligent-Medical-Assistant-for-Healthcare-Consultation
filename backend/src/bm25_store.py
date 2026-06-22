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
        self.index: Any = None
        self.documents: list[str] = []
        self.doc_ids: list[str] = []
        self.metadatas: list[dict[str, Any]] = []
        self.tokenized: list[list[str]] = []
        self._idf: dict[str, float] = {}

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        docs = [self._tokenize(chunk["content"]) for chunk in chunks]
        self.tokenized = docs
        self.documents = [chunk["content"] for chunk in chunks]
        self.doc_ids = [str(chunk["id"]) for chunk in chunks]
        self.metadatas = [self._metadata(chunk) for chunk in chunks]
        if BM25Okapi is not None:
            self.index = BM25Okapi(docs)
        else:
            self.index = "simple"
            self._idf = self._build_simple_idf(docs)

    def search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        if self.index is None:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        if BM25Okapi is not None and self.index != "simple":
            raw_scores = self.index.get_scores(query_tokens)
            scores = [float(score) for score in raw_scores]
        else:
            scores = self._simple_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )
        results = []
        for index in ranked_indices:
            if len(results) >= top_k:
                break
            if scores[index] <= 0:
                continue
            metadata = self.metadatas[index]
            if category_filter and metadata.get("category") != category_filter:
                continue
            results.append(
                {
                    "id": self.doc_ids[index],
                    "doc_id": self.doc_ids[index],
                    "content": self.documents[index],
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
                    "index": self.index,
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
        self.index = data.get("index")
        self.documents = data.get("documents", [])
        self.doc_ids = data.get("doc_ids", [])
        self.metadatas = data.get("metadatas", [])
        self.tokenized = data.get("tokenized", [])
        self._idf = data.get("idf", {})
        return True

    def _tokenize(self, text: str) -> list[str]:
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
            "language": "vi",
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

    def _simple_scores(self, query_tokens: list[str]) -> list[float]:
        idf = self._idf
        scores = []
        for doc in self.tokenized:
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
