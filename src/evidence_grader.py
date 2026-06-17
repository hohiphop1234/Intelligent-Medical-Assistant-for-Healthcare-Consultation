from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import (
    EVIDENCE_THRESHOLD,
    LLM_MODEL,
    MIN_EVIDENCE_CHUNKS,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from src.utils import normalize_for_match, tokenize

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


@dataclass
class GradingResult:
    relevant_chunks: list[dict[str, Any]]
    score: float
    needs_crawl: bool
    confidence: str


class EvidenceGrader:
    """Grade whether retrieved chunks are useful enough to answer."""

    def __init__(self, use_llm: bool = True):
        self.client = None
        if (
            use_llm
            and OpenAI is not None
            and OPENROUTER_API_KEY
            and OPENROUTER_API_KEY != "your_key_here"
        ):
            self.client = OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
            )

    def grade(self, question: str, chunks: list[dict[str, Any]]) -> GradingResult:
        relevant = []
        for chunk in chunks:
            relevance_score = self._score_single(question, chunk)
            chunk = {**chunk, "relevance_score": relevance_score}
            if relevance_score >= EVIDENCE_THRESHOLD:
                relevant.append(chunk)

        if not relevant and chunks:
            # Keep the best weak evidence visible to the caller, but still request crawl.
            best = max(chunks, key=lambda item: item.get("score", item.get("fused_score", 0)))
            best_score = self._score_single(question, best)
            if best_score >= 0.25:
                relevant.append({**best, "relevance_score": best_score})

        average = sum(item.get("relevance_score", 0.0) for item in relevant) / max(
            len(relevant), 1
        )
        needs_crawl = len(relevant) < MIN_EVIDENCE_CHUNKS
        confidence = "high" if average >= 0.75 else "medium" if average >= 0.45 else "low"
        return GradingResult(
            relevant_chunks=relevant,
            score=average,
            needs_crawl=needs_crawl,
            confidence=confidence,
        )

    def _score_single(self, question: str, chunk: dict[str, Any]) -> float:
        if self.client is not None:
            try:
                return 1.0 if self._grade_single_llm(question, chunk["content"]) else 0.0
            except Exception:
                pass
        return self._grade_single_rules(question, chunk)

    def _grade_single_llm(self, question: str, chunk_content: str) -> bool:
        prompt = (
            f"Question: {question}\n\n"
            f"Document:\n{chunk_content[:800]}\n\n"
            "Is this document relevant and useful for answering the question? "
            "Answer only relevant or irrelevant."
        )
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        answer = (response.choices[0].message.content or "").lower()
        return "relevant" in answer and "irrelevant" not in answer

    def _grade_single_rules(self, question: str, chunk: dict[str, Any]) -> float:
        q_tokens = set(tokenize(question))
        c_tokens = set(tokenize(chunk.get("content", "")))
        if not q_tokens or not c_tokens:
            return 0.0

        query_entities = self._query_entities(question)
        overlap = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
        metadata = chunk.get("metadata", {})
        entity = normalize_for_match(str(metadata.get("entity") or chunk.get("entity", "")))
        question_normalized = normalize_for_match(question)
        content_normalized = normalize_for_match(chunk.get("content", ""))
        entity_matches_query = bool(entity and entity in question_normalized)
        content_has_query_entity = any(entity_name in content_normalized for entity_name in query_entities)
        if query_entities and not entity_matches_query and not content_has_query_entity:
            return min(0.2, overlap * 0.4)

        entity_bonus = 0.3 if entity_matches_query or content_has_query_entity else 0.0
        search_score = float(
            chunk.get("score", chunk.get("vector_score", chunk.get("fused_score", 0.0))) or 0.0
        )
        search_bonus = min(max(search_score, 0.0), 1.0) * 0.2
        return min(1.0, overlap * 0.7 + entity_bonus + search_bonus)

    def _query_entities(self, question: str) -> set[str]:
        normalized = normalize_for_match(question)
        known = {
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
        }
        return {entity for entity in known if entity in normalized}
