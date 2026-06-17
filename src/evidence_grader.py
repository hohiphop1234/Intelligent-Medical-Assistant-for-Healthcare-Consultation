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
from src.topic_relevance import pregnancy_hard_reject, pregnancy_relevance_bonus
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

        relevant.sort(key=lambda item: item.get("relevance_score", 0.0), reverse=True)

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
        if self._entity_mismatch(question, chunk):
            return 0.0
        if pregnancy_hard_reject(
            question, chunk.get("content", ""), chunk.get("metadata", {})
        ):
            return 0.0
        rule_score = self._grade_single_rules(question, chunk)
        if self.client is not None:
            try:
                if self._grade_single_llm(question, chunk["content"]):
                    return max(rule_score, 0.8)
                return min(rule_score, 0.2)
            except Exception:
                pass
        return rule_score

    def _grade_single_llm(self, question: str, chunk_content: str) -> bool:
        prompt = (
            f"Question: {question}\n\n"
            f"Document:\n{chunk_content[:800]}\n\n"
            "Is this document relevant and useful for answering the question? "
            "For pregnancy questions, mark breastfeeding/lactation/postpartum/milk-only "
            "documents irrelevant unless the question asks about breastfeeding. "
            "For questions asking what medicines to avoid in pregnancy, the document "
            "must directly discuss pregnancy, fetal/unborn-baby risk, contraindication, "
            "or avoiding a medicine during pregnancy. "
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
        search_score = float(
            chunk.get("score", chunk.get("vector_score", chunk.get("fused_score", 0.0))) or 0.0
        )
        if self._is_interaction_question(question, query_entities):
            return self._interaction_score(
                question_normalized,
                content_normalized,
                metadata,
                query_entities,
                overlap,
                search_score,
            )

        entity_matches_query = bool(entity and entity in question_normalized)
        content_has_query_entity = any(entity_name in content_normalized for entity_name in query_entities)
        if query_entities and not entity_matches_query and not content_has_query_entity:
            return min(0.2, overlap * 0.4)

        entity_bonus = 0.3 if entity_matches_query or content_has_query_entity else 0.0
        topic_bonus = pregnancy_relevance_bonus(question, chunk.get("content", ""), metadata)
        search_bonus = min(max(search_score, 0.0), 1.0) * 0.2
        return min(1.0, overlap * 0.7 + entity_bonus + topic_bonus + search_bonus)

    def _is_interaction_question(self, question: str, query_entities: set[str]) -> bool:
        normalized = normalize_for_match(question)
        interaction_terms = [
            "interaction",
            "interact",
            "take with",
            "with",
            "together",
            "combine",
            "mix",
            "tuong tac",
            "dung chung",
            "uong chung",
            "ket hop",
            "voi",
        ]
        return len(query_entities) >= 2 and any(term in normalized for term in interaction_terms)

    def _interaction_score(
        self,
        question_normalized: str,
        content_normalized: str,
        metadata: dict[str, Any],
        query_entities: set[str],
        overlap: float,
        search_score: float,
    ) -> float:
        interaction_cues = [
            "interact",
            "interaction",
            "nsaid",
            "nonsteroidal",
            "anti inflammatory",
            "blood thinner",
            "anticoagulant",
            "bleeding risk",
            "risk of bleeding",
            "increase the risk of bleeding",
            "do not start",
            "without discussing",
            "closely monitor",
            "monitor inr",
            "concomitant",
            "tuong tac",
            "chay mau",
            "nguy co chay mau",
            "khong bat dau",
        ]
        if not any(cue in content_normalized for cue in interaction_cues):
            return min(0.2, overlap * 0.5)

        entity_hits = {entity for entity in query_entities if entity in content_normalized}
        metadata_entity = normalize_for_match(str(metadata.get("entity", "")))
        if metadata_entity in query_entities:
            entity_hits.add(metadata_entity)

        nsaid_support = "ibuprofen" in query_entities and (
            "nsaid" in content_normalized
            or "nonsteroidal" in content_normalized
            or "anti inflammatory" in content_normalized
        )
        if not entity_hits and not nsaid_support:
            return 0.15

        entity_score = len(entity_hits) / max(len(query_entities), 1)
        if nsaid_support:
            entity_score = max(entity_score, 0.75)

        warning_bonus = 0.0
        if any(
            cue in content_normalized
            for cue in [
                "do not start",
                "without discussing",
                "closely monitor",
                "tell your doctor",
                "tell your healthcare provider",
                "khong bat dau",
                "hoi bac si",
            ]
        ):
            warning_bonus += 0.15
        if "warfarin" in question_normalized and (
            "bleeding risk" in content_normalized
            or "risk of bleeding" in content_normalized
            or "increase the risk of bleeding" in content_normalized
        ):
            warning_bonus += 0.15

        category = str(metadata.get("category", ""))
        category_penalty = 0.25 if category == "pregnancy" else 0.0
        search_bonus = min(max(search_score, 0.0), 1.0) * 0.15
        score = 0.3 + (entity_score * 0.3) + (overlap * 0.3) + warning_bonus + search_bonus
        return min(1.0, max(0.0, score - category_penalty))

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

    def _entity_mismatch(self, question: str, chunk: dict[str, Any]) -> bool:
        query_entities = self._query_entities(question)
        if not query_entities:
            return False
        metadata = chunk.get("metadata", {})
        entity = normalize_for_match(str(metadata.get("entity") or chunk.get("entity", "")))
        content = normalize_for_match(chunk.get("content", ""))
        return not (
            entity in query_entities
            or any(entity_name in content for entity_name in query_entities)
        )
