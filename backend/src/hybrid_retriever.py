from __future__ import annotations

from typing import Any

from config import BM25_WEIGHT, TOP_K, VECTOR_WEIGHT
from src.topic_relevance import pregnancy_hard_reject, pregnancy_relevance_bonus
from src.utils import expand_query_with_drug_aliases, extract_drug_entities, normalize_for_match


class HybridRetriever:
    """Hybrid search using vector results, BM25 results, and RRF fusion."""

    def __init__(self, embedding_manager, vector_store, bm25_store):
        self.embeddings = embedding_manager
        self.vector_store = vector_store
        self.bm25 = bm25_store

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        category_filter: str | None = None,
    ) -> list[dict[str, Any]]:

        expanded_query = expand_query_with_drug_aliases(query)
        query_embedding = self.embeddings.embed(expanded_query)
        vector_results = self.vector_store.search(
            query_embedding,
            top_k=top_k * 2,
            category_filter=category_filter,
        )
        bm25_results = self.bm25.search(
            expanded_query, top_k=top_k * 2, category_filter=category_filter
        )
        fused = self._rrf_fusion(vector_results, bm25_results)
        return self._rerank_for_query(expanded_query, fused)[:top_k]

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

    def _rerank_for_query(
        self, query: str, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        normalized_query = normalize_for_match(query)
        query_entities = set(
            extract_drug_entities(
                normalized_query,
                [
                "warfarin",
                "ibuprofen",
                "acetaminophen",
                "paracetamol",
                "metformin",
                "insulin",
                "aspirin",
                "omeprazole",
                "famotidine",
                "phenytoin",
                "panadol",
                "tylenol",
                ],
            )
        )
        asks_side_effects = any(
            phrase in normalized_query
            for phrase in ["side effect", "side effects", "adverse effect", "tac dung phu"]
        )
        side_effect_cues = [
            "tac dung phu",
            "neu ban gap",
            "goi bac si",
            "trieu chung",
            "side effects",
            "if you experience",
            "call your doctor",
            "symptoms",
        ]

        reranked = []
        for result in results:
            metadata = result.get("metadata", {})
            entity = normalize_for_match(str(metadata.get("entity", "")))
            content = normalize_for_match(result.get("content", ""))
            boost = 0.0
            if query_entities:
                if entity in query_entities or any(e in content for e in query_entities):
                    boost += 0.12
                else:
                    boost -= 0.25
            if asks_side_effects and any(cue in content for cue in side_effect_cues):
                boost += 0.01
            if pregnancy_hard_reject(query, result.get("content", ""), metadata):
                boost -= 0.08
            else:
                boost += pregnancy_relevance_bonus(
                    query, result.get("content", ""), metadata
                )
            reranked.append({**result, "rerank_score": result.get("fused_score", 0.0) + boost})

        return sorted(reranked, key=lambda item: item.get("rerank_score", 0.0), reverse=True)
