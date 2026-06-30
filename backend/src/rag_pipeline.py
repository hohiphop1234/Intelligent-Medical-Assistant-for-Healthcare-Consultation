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

    def process_query(self, question: str, isEmergency: bool = False) -> dict[str, Any]:
        if isEmergency:
            return self.process_emergency_query(question)

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

    def stream_query(self, question: str, isEmergency: bool = False):
        if isEmergency:
            for item in self.stream_emergency_query(question):
                yield item
            return

        classification = self.query_router.classify(question)
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

    def process_emergency_query(self, question: str) -> dict[str, Any]:
        """Quy trình RAG chuyên sâu cho nhánh khẩn cấp / cấp cứu"""
        from src.query_router import QueryClassification
        classification = QueryClassification(
            intent="emergency_triage",
            category="overdose_triage",
            entities=[],
            risk_level="critical",
            confidence=1.0,
            requires_rag=True
        )
        results = self.hybrid_retriever.search(
            question,
            top_k=TOP_K + 2,
            category_filter=None,
        )
        graded = self.evidence_grader.grade(question, results)
        if not graded.relevant_chunks and results:
            graded.relevant_chunks = results[:2]

        if not graded.relevant_chunks:
            resp = self.safety_guard.emergency_response(question)
            return {
                "type": "emergency",
                "answer": resp["message"],
                "sources": [],
                "evidence_score": 0.0,
                "retrieved_count": 0,
                "relevant_count": 0,
                "route": "emergency_rag",
                "risk_level": "critical",
                "category": "overdose_triage"
            }

        response = self.response_generator.generate(
            f"[TÌNH HUỐNG KHẨN CẤP Y TẾ - HÃY TRẢ LỜI NGẮN GỌN CÁC BƯỚC SƠ CỨU AN TOÀN] {question}",
            graded.relevant_chunks,
            classification
        )
        urgent_prefix = (
            "**🚨 CẢNH BÁO KHẨN CẤP:** Tình huống bạn mô tả có thể là cấp cứu y tế nghiêm trọng. "
            "Hãy gọi ngay **115** hoặc đến cơ sở y tế gần nhất.\n\n"
            "--- \n"
            "**Hướng dẫn sơ cứu ban đầu từ cơ sở dữ liệu y khoa:**\n\n"
        )
        answer = urgent_prefix + response.get("answer", "")
        return {
            "type": "emergency",
            "answer": answer,
            "sources": response.get("sources", []),
            "evidence_score": graded.score,
            "retrieved_count": len(results),
            "relevant_count": len(graded.relevant_chunks),
            "route": "emergency_rag",
            "risk_level": "critical",
            "category": "overdose_triage"
        }

    def stream_emergency_query(self, question: str):
        """Stream trả lời RAG cho nhánh khẩn cấp"""
        from src.query_router import QueryClassification
        classification = QueryClassification(
            intent="emergency_triage",
            category="overdose_triage",
            entities=[],
            risk_level="critical",
            confidence=1.0,
            requires_rag=True
        )
        results = self.hybrid_retriever.search(
            question,
            top_k=TOP_K + 2,
            category_filter=None,
        )
        graded = self.evidence_grader.grade(question, results)
        if not graded.relevant_chunks and results:
            graded.relevant_chunks = results[:2]

        if not graded.relevant_chunks:
            resp = self.safety_guard.emergency_response(question)
            yield {"type": "metadata", "data": {"type": "emergency", "message": resp["message"]}}
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
        disclaimer = "**🚨 CẢNH BÁO KHẨN CẤP:** Hãy gọi ngay 115 hoặc đến cơ sở y tế gần nhất. Thông tin sơ cứu trên chỉ hỗ trợ ban đầu và không thay thế bác sĩ."
        yield {
            "type": "metadata",
            "data": {
                "type": "message",
                "sources": sources,
                "category": "overdose_triage",
                "risk_level": "critical",
                "route": "emergency_rag",
                "disclaimer": disclaimer
            }
        }
        urgent_prefix = (
            "**🚨 CẢNH BÁO KHẨN CẤP:** Tình huống bạn mô tả có thể là cấp cứu y tế nghiêm trọng. "
            "Hãy gọi ngay **115** hoặc đến cơ sở y tế gần nhất.\n\n"
            "--- \n"
            "**Hướng dẫn sơ cứu ban đầu từ cơ sở dữ liệu y khoa:**\n\n"
        )
        yield {"type": "token", "content": urgent_prefix}
        for token in self.response_generator.generate_stream(
            f"[TÌNH HUỐNG KHẨN CẤP Y TẾ - HÃY TRẢ LỜI NGẮN GỌN CÁC BƯỚC SƠ CỨU AN TOÀN] {question}",
            graded.relevant_chunks,
            classification
        ):
            yield {"type": "token", "content": token}

    def _retrieval_category_filter(
        self, question: str, classification
    ) -> str | None:
        # Remove hard category filtering to avoid excluding relevant but miscategorized chunks
        return None

    def _retrieval_top_k(self, question: str, classification) -> int:
        if classification.category == "drug_interaction":
            return TOP_K * 3 + 2
        if (
            classification.category == "pregnancy"
            and should_route_pregnancy_to_drug_safety(question)
            and asks_drug_avoidance(question)
            and not classification.entities
        ):
            return TOP_K * 4 + 2
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
