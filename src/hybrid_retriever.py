from __future__ import annotations

from typing import Any

from config import BM25_WEIGHT, TOP_K, VECTOR_WEIGHT


class HybridRetriever:
    """Hybrid search using vector results, BM25 results, and RRF fusion."""

    def __init__(self, embedding_manager, vector_store, bm25_store):
        self.embeddings = embedding_manager
        self.vector_store = vector_store
        self.bm25 = bm25_store

    def search(
        self,
        query: str,
        language: str | None = None,
        top_k: int = TOP_K,
        category_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        if language is None:
            language = self.embeddings.detect_language(query)

        query_embedding = self.embeddings.embed(query, language)
        vector_results = self.vector_store.search(
            query_embedding,
            language,
            top_k=top_k * 2,
            category_filter=category_filter,
        )
        bm25_results = self.bm25.search(
            query, language, top_k=top_k * 2, category_filter=category_filter
        )
        return self._rrf_fusion(vector_results, bm25_results)[:top_k]

    def _rrf_fusion(
        self,
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = {}
        doc_map: dict[str, dict[str, Any]] = {}

        for rank, doc in enumerate(vector_results):
            doc_id = self._doc_id(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + VECTOR_WEIGHT / (k + rank + 1)
            merged = {**doc, "vector_score": doc.get("score", 0.0)}
            doc_map[doc_id] = merged

        for rank, doc in enumerate(bm25_results):
            doc_id = self._doc_id(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + BM25_WEIGHT / (k + rank + 1)
            if doc_id in doc_map:
                doc_map[doc_id]["bm25_score"] = doc.get("score", 0.0)
                doc_map[doc_id]["metadata"] = {
                    **doc.get("metadata", {}),
                    **doc_map[doc_id].get("metadata", {}),
                }
            else:
                doc_map[doc_id] = {**doc, "bm25_score": doc.get("score", 0.0)}

        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        fused = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id]
            fused.append({**doc, "fused_score": scores[doc_id]})
        return fused

    def _doc_id(self, doc: dict[str, Any]) -> str:
        return str(doc.get("id") or doc.get("doc_id") or doc.get("content", "")[:80])
