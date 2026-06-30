from __future__ import annotations

from pathlib import Path
from typing import Any

from config import (
    CATEGORIES_PATH,
    CHROMA_PERSIST_DIR,
    CONFIDENCE_THRESHOLD,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TOP_K,
)
from src.bm25_store import BM25Store
from src.data_cleaner import DataCleaner
from src.embeddings import EmbeddingManager
from src.evidence_grader import EvidenceGrader
from src.hybrid_retriever import HybridRetriever
from src.query_router import QueryRouter
from src.response_generator import ResponseGenerator
from src.response_validator import ResponseValidator
from src.safety_guard import SafetyGuard
from src.utils import ensure_dir, load_jsonl
from src.vector_store import VectorStore
from src.web_crawler import WebCrawler


BM25_INDEX_PATH = str(Path("models") / "bm25_index.pkl")
FILTERABLE_CATEGORIES = {
    "drug_safety",
    "disease_knowledge",
    "overdose_triage",
    "pregnancy",
    "pediatric",
    "elderly",
}


class MedicalRAGPipeline:
    """End-to-end medical RAG flow."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.safety_guard = SafetyGuard(llm_client=self.llm_client)
        self.query_router = QueryRouter(llm_client=self.llm_client)
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore(CHROMA_PERSIST_DIR)
        self.bm25_store = BM25Store()
        self.bm25_store.load(BM25_INDEX_PATH)
        self.hybrid_retriever = HybridRetriever(
            self.embedding_manager, self.vector_store, self.bm25_store
        )
        self.evidence_grader = EvidenceGrader()
        self.web_crawler = WebCrawler()
        self.response_generator = ResponseGenerator()
        self.response_validator = ResponseValidator()

    def process_query(self, question: str) -> dict[str, Any]:
        if self.safety_guard.is_emergency(question):
            return self.safety_guard.emergency_response(question)

        classification = self.query_router.classify(question)
        if classification.category == "out_of_scope":
            return self.safety_guard.out_of_scope_response(question)

        if classification.confidence < CONFIDENCE_THRESHOLD:
            return self.safety_guard.insufficient_evidence_response(question)

        category_filter = self._retrieval_category_filter(question, classification)
        search_top_k = self._retrieval_top_k(question, classification)
        results = self.hybrid_retriever.search(
            question,
            top_k=search_top_k,
            category_filter=category_filter,
        )

        graded = self.evidence_grader.grade(question, results)
        if graded.needs_crawl:
            crawled = self.web_crawler.search(question, classification.entities)
            if crawled:
                for chunk in crawled:
                    chunk["embedding"] = self.embedding_manager.embed(
                        chunk["content"]
                    )
                graded = self.evidence_grader.grade(question, results + crawled)

        if not graded.relevant_chunks:
            return self.safety_guard.insufficient_evidence_response(question)

        response = self.response_generator.generate(
            question, graded.relevant_chunks, classification
        )
        response["confidence"] = graded.confidence
        response["evidence_score"] = graded.score
        response["retrieved_count"] = len(results)
        response["relevant_count"] = len(graded.relevant_chunks)
        return self.response_validator.validate(response, graded.relevant_chunks)

    def stream_query(self, question: str, is_emergency: bool = False):
        if is_emergency or self.safety_guard.is_emergency(question):
            yield {"type": "metadata", "data": {"type": "emergency", "message": self.safety_guard.emergency_response(question)["message"]}}
            return

        classification = self.query_router.classify(question)
        if classification.category == "faq":
            yield {"type": "metadata", "data": {"type": "message", "sources": [], "category": "faq", "risk_level": "low", "route": "general_qa"}}
            for token in self.llm_client.stream(question):
                yield {"type": "token", "content": token}
            return
        if classification.category == "out_of_scope":
            yield {"type": "metadata", "data": {"type": "out_of_scope", "message": self.safety_guard.out_of_scope_response(question)["message"]}}
            return

        if classification.confidence < CONFIDENCE_THRESHOLD:
            yield {"type": "metadata", "data": {"type": "insufficient_evidence", "message": self.safety_guard.insufficient_evidence_response(question)["message"]}}
            return

        category_filter = self._retrieval_category_filter(question, classification)
        search_top_k = self._retrieval_top_k(question, classification)
        results = self.hybrid_retriever.search(
            question,
            top_k=search_top_k,
            category_filter=category_filter,
        )

        graded = self.evidence_grader.grade(question, results)
        
        if not graded.relevant_chunks:
            yield {"type": "metadata", "data": {"type": "insufficient_evidence", "message": self.safety_guard.insufficient_evidence_response(question)["message"]}}
            return

        sources = [
            {
                "index": i + 1,
                "id": chunk.get("id", ""),
                "title": chunk.get("title", ""),
                "content": chunk.get("content", "")
            }
            for i, chunk in enumerate(graded.relevant_chunks)
        ]
        
        from src.safety_guard import get_disclaimer
        yield {
            "type": "metadata",
            "data": {
                "type": "message",
                "sources": sources,
                "category": classification.category,
                "risk_level": classification.risk_level,
                "route": "rag",
                "disclaimer": get_disclaimer(classification.risk_level)
            }
        }
        
        for token in self.response_generator.generate_stream(question, graded.relevant_chunks, classification):
            yield {"type": "token", "content": token}

    def _retrieval_category_filter(
        self, question: str, classification
    ) -> str | None:
        # Remove hard category filtering to avoid excluding relevant but miscategorized chunks
        return None

    def _retrieval_top_k(self, question: str, classification) -> int:
        if classification.category == "drug_interaction":
            return TOP_K * 3 + 2
        return TOP_K + 2

    def ingest_data(self) -> dict[str, int]:
        ensure_dir(PROCESSED_DATA_DIR)
        ensure_dir(Path(BM25_INDEX_PATH).parent)
        cleaner = DataCleaner()
        cleaner.process_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR)
        self.vector_store.reset()

        all_chunks: dict[str, list[dict[str, Any]]] = {}
        chunks_path = Path(PROCESSED_DATA_DIR) / "chunks_vi.jsonl"
        chunks = load_jsonl(chunks_path)
        all_chunks["vi"] = chunks
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_manager.embed_batch(texts)
        self.vector_store.add_documents(chunks, embeddings)
        self.bm25_store.build_index(chunks)

        self.bm25_store.save(BM25_INDEX_PATH)
        stats = self.vector_store.get_stats()
        stats["bm25_vi_count"] = len(all_chunks.get("vi", []))
        return stats
