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
from src.embeddings import DualEmbeddingManager
from src.evidence_grader import EvidenceGrader
from src.hybrid_retriever import HybridRetriever
from src.query_router import QueryRouter
from src.response_generator import ResponseGenerator
from src.response_validator import ResponseValidator
from src.safety_guard import SafetyGuard
from src.topic_relevance import asks_drug_avoidance, should_route_pregnancy_to_drug_safety
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

    def __init__(self):
        self.safety_guard = SafetyGuard(CATEGORIES_PATH)
        self.query_router = QueryRouter(CATEGORIES_PATH)
        self.embedding_manager = DualEmbeddingManager()
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
        language = self.embedding_manager.detect_language(question)

        if self.safety_guard.is_emergency(question):
            return self.safety_guard.emergency_response(question, language)

        classification = self.query_router.classify(question, language)
        if classification.category == "out_of_scope":
            return self.safety_guard.out_of_scope_response(question, language)

        if classification.confidence < CONFIDENCE_THRESHOLD:
            return self.safety_guard.insufficient_evidence_response(question, language)

        category_filter = self._retrieval_category_filter(question, classification)
        search_top_k = self._retrieval_top_k(question, classification)
        results = self.hybrid_retriever.search(
            question,
            language=language,
            top_k=search_top_k,
            category_filter=category_filter,
        )

        graded = self.evidence_grader.grade(question, results)
        if graded.needs_crawl:
            crawled = self.web_crawler.search(question, classification.entities)
            if crawled:
                for chunk in crawled:
                    chunk["embedding"] = self.embedding_manager.embed(
                        chunk["content"], language
                    )
                graded = self.evidence_grader.grade(question, results + crawled)

        if not graded.relevant_chunks:
            return self.safety_guard.insufficient_evidence_response(question, language)

        response = self.response_generator.generate(
            question, graded.relevant_chunks, classification
        )
        response["confidence"] = graded.confidence
        response["evidence_score"] = graded.score
        response["retrieved_count"] = len(results)
        response["relevant_count"] = len(graded.relevant_chunks)
        return self.response_validator.validate(response, graded.relevant_chunks)

    def _retrieval_category_filter(
        self, question: str, classification
    ) -> str | None:
        if classification.category == "pregnancy" and should_route_pregnancy_to_drug_safety(
            question
        ):
            return "drug_safety"
        return (
            classification.category
            if classification.category in FILTERABLE_CATEGORIES
            else None
        )

    def _retrieval_top_k(self, question: str, classification) -> int:
        if (
            classification.category == "pregnancy"
            and should_route_pregnancy_to_drug_safety(question)
            and asks_drug_avoidance(question)
            and not classification.entities
        ):
            return TOP_K * 4
        return TOP_K

    def ingest_data(self) -> dict[str, int]:
        ensure_dir(PROCESSED_DATA_DIR)
        ensure_dir(Path(BM25_INDEX_PATH).parent)
        cleaner = DataCleaner()
        cleaner.process_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR)
        self.vector_store.reset()

        all_chunks: dict[str, list[dict[str, Any]]] = {}
        for language in ["en", "vi"]:
            chunks_path = Path(PROCESSED_DATA_DIR) / f"chunks_{language}.jsonl"
            chunks = load_jsonl(chunks_path)
            all_chunks[language] = chunks
            texts = [chunk["content"] for chunk in chunks]
            embeddings = self.embedding_manager.embed_batch(texts, language=language)
            self.vector_store.add_documents(chunks, embeddings, language=language)
            self.bm25_store.build_index(chunks, language=language)

        self.bm25_store.save(BM25_INDEX_PATH)
        stats = self.vector_store.get_stats()
        stats["bm25_vi_count"] = len(all_chunks.get("vi", []))
        stats["bm25_en_count"] = len(all_chunks.get("en", []))
        return stats
