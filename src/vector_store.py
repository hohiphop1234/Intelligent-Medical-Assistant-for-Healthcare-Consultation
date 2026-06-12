from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import COLLECTION_NAME_EN, COLLECTION_NAME_VI
from src.utils import cosine_similarity, ensure_dir

try:
    import chromadb
except ImportError:  # pragma: no cover - optional dependency
    chromadb = None


class VectorStore:
    """Chroma-backed vector store with a local JSON fallback."""

    def __init__(self, persist_dir: str, use_chroma: bool | None = None):
        self.persist_dir = ensure_dir(persist_dir)
        self.use_chroma = chromadb is not None if use_chroma is None else use_chroma
        if self.use_chroma and chromadb is None:
            self.use_chroma = False
        self.collections: dict[str, Any] = {}
        self._fallback_docs: dict[str, dict[str, dict[str, Any]]] = {"vi": {}, "en": {}}

        if self.use_chroma:
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collections["vi"] = self.client.get_or_create_collection(
                name=COLLECTION_NAME_VI, metadata={"hnsw:space": "cosine"}
            )
            self.collections["en"] = self.client.get_or_create_collection(
                name=COLLECTION_NAME_EN, metadata={"hnsw:space": "cosine"}
            )
        else:
            self._load_fallback()

    def add_documents(
        self, chunks: list[dict[str, Any]], embeddings: list[list[float]], language: str
    ) -> None:
        if not chunks:
            return
        if self.use_chroma:
            collection = self.collections[language]
            collection.upsert(
                ids=[str(chunk["id"]) for chunk in chunks],
                documents=[chunk["content"] for chunk in chunks],
                embeddings=embeddings,
                metadatas=[self._metadata(chunk) for chunk in chunks],
            )
            return

        docs = self._fallback_docs[language]
        for chunk, embedding in zip(chunks, embeddings):
            docs[str(chunk["id"])] = {
                "content": chunk["content"],
                "embedding": embedding,
                "metadata": self._metadata(chunk),
            }
        self._save_fallback(language)

    def reset(self) -> None:
        """Clear persisted vector collections before a full re-ingest."""
        if self.use_chroma:
            for language, name in {
                "vi": COLLECTION_NAME_VI,
                "en": COLLECTION_NAME_EN,
            }.items():
                try:
                    self.client.delete_collection(name)
                except Exception:
                    pass
                self.collections[language] = self.client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"},
                )
            return

        self._fallback_docs = {"vi": {}, "en": {}}
        for language in ["vi", "en"]:
            path = self._fallback_path(language)
            if path.exists():
                path.unlink()

    def search(
        self,
        query_embedding: list[float],
        language: str,
        top_k: int = 5,
        category_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        if self.use_chroma:
            collection = self.collections[language]
            if collection.count() == 0:
                return []
            where_filter = {"category": category_filter} if category_filter else None
            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_filter,
                    include=["documents", "distances", "metadatas"],
                )
            except Exception:
                if where_filter is None:
                    raise
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "distances", "metadatas"],
                )
            return self._format_chroma_results(results)

        scored = []
        for doc_id, doc in self._fallback_docs[language].items():
            metadata = doc.get("metadata", {})
            if category_filter and metadata.get("category") != category_filter:
                continue
            score = cosine_similarity(query_embedding, doc["embedding"])
            scored.append(
                {
                    "id": doc_id,
                    "content": doc["content"],
                    "score": score,
                    "metadata": metadata,
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def get_stats(self) -> dict[str, int]:
        if self.use_chroma:
            return {
                "vi_count": self.collections["vi"].count(),
                "en_count": self.collections["en"].count(),
            }
        return {
            "vi_count": len(self._fallback_docs["vi"]),
            "en_count": len(self._fallback_docs["en"]),
        }

    def _format_chroma_results(self, results: dict[str, Any]) -> list[dict[str, Any]]:
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        formatted = []
        for doc_id, document, distance, metadata in zip(
            ids, documents, distances, metadatas
        ):
            formatted.append(
                {
                    "id": doc_id,
                    "content": document,
                    "score": max(0.0, 1.0 - float(distance)),
                    "metadata": metadata or {},
                }
            )
        return formatted

    def _metadata(self, chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
        risk_categories = chunk.get("supported_risk_categories") or []
        if isinstance(risk_categories, list):
            risk_categories_value = ",".join(str(item) for item in risk_categories)
        else:
            risk_categories_value = str(risk_categories)
        return {
            "source": str(chunk.get("source", "")),
            "entity": str(chunk.get("entity", "")),
            "category": str(chunk.get("category", chunk.get("topic_group", ""))),
            "topic_group": str(chunk.get("topic_group", "")),
            "risk_level": str(chunk.get("risk_level", "medium")),
            "section": str(chunk.get("section", "")),
            "url": str(chunk.get("url", "")),
            "title": str(chunk.get("title", "")),
            "language": str(chunk.get("language", "")),
            "risk_categories": risk_categories_value,
            "word_count": int(chunk.get("word_count", 0) or 0),
            "quality_score": float(chunk.get("quality_score", 0.0) or 0.0),
        }

    def _fallback_path(self, language: str) -> Path:
        return self.persist_dir / f"simple_{language}.json"

    def _load_fallback(self) -> None:
        for language in ["vi", "en"]:
            path = self._fallback_path(language)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self._fallback_docs[language] = json.load(f)

    def _save_fallback(self, language: str) -> None:
        path = self._fallback_path(language)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._fallback_docs[language], f, ensure_ascii=False)
