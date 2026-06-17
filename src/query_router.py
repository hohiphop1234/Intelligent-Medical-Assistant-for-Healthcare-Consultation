from __future__ import annotations

import json
from dataclasses import dataclass

from config import (
    CATEGORIES_PATH,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from src.safety_guard import SafetyGuard
from src.utils import extract_drug_entities, normalize_for_match, safe_json_loads

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


KNOWN_DRUGS = [
    "warfarin",
    "ibuprofen",
    "acetaminophen",
    "paracetamol",
    "metformin",
    "insulin",
    "aspirin",
    "omeprazole",
    "famotidine",
    "panadol",
    "tylenol",
]


@dataclass
class QueryClassification:
    intent: str
    category: str
    entities: list[str]
    risk_level: str
    confidence: float
    requires_rag: bool
    language: str


class QueryRouter:
    """Classify medical queries before retrieval."""

    def __init__(self, categories_path: str = CATEGORIES_PATH):
        self.safety_guard = SafetyGuard(categories_path)
        self.client = None
        if (
            OpenAI is not None
            and OPENROUTER_API_KEY
            and OPENROUTER_API_KEY != "your_key_here"
        ):
            self.client = OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
            )

    def classify(self, query: str, language: str) -> QueryClassification:
        if self.client is not None:
            try:
                return self._classify_with_llm(query, language)
            except Exception:
                pass
        return self._classify_with_rules(query, language)

    def _classify_with_llm(self, query: str, language: str) -> QueryClassification:
        prompt = f"""Classify this user query for a medical RAG assistant.
Return JSON only.

Categories:
drug_safety, drug_interaction, overdose_triage, disease_knowledge,
pregnancy, pediatric, elderly, general_health, out_of_scope

Rules:
- All medical questions require RAG.
- If the query is not medical, category must be out_of_scope.
- Use critical risk for overdose, emergency, pregnancy, pediatric, and interactions.

Query: {json.dumps(query)}

Return:
{{"intent":"...","category":"...","entities":["..."],"risk_level":"low|medium|high|critical","confidence":0.0,"requires_rag":true}}"""
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict medical query classifier. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=min(LLM_MAX_TOKENS, 300),
        )
        content = response.choices[0].message.content or "{}"
        data = safe_json_loads(content) or {}
        data["language"] = language
        data["requires_rag"] = data.get("category") != "out_of_scope"
        classification = self._coerce_classification(data, language)
        alias_entities = self._extract_entities(normalize_for_match(query))
        if alias_entities:
            classification.entities = sorted(set(classification.entities) | set(alias_entities))
            classification.requires_rag = True
            if classification.category == "out_of_scope":
                classification.category = "drug_safety"
                classification.intent = "drug_info"
                classification.risk_level = "high"
                classification.confidence = max(classification.confidence, 0.82)
        return classification

    def _classify_with_rules(self, query: str, language: str) -> QueryClassification:
        q = normalize_for_match(query)
        entities = self._extract_entities(q)

        category = "out_of_scope"
        intent = "out_of_scope"
        risk = "none"
        confidence = 0.35
        requires_rag = False

        scoped, scope_score = self.safety_guard.is_medical_scope(query)
        has_known_drug = bool(entities)

        if any(term in q for term in ["qua lieu", "overdose", "poison", "ngo doc", "too many"]):
            category, intent, risk, confidence = (
                "overdose_triage",
                "emergency_or_overdose",
                "critical",
                0.92,
            )
        elif len(entities) >= 2 and any(term in q.split() for term in ["with", "voi", "chung"]):
            category, intent, risk, confidence = (
                "drug_interaction",
                "interaction_check",
                "critical",
                0.88,
            )
        elif any(
            term in q
            for term in ["tuong tac", "interaction", "interact", "dung chung", "take with", "ket hop"]
        ):
            category, intent, risk, confidence = (
                "drug_interaction",
                "interaction_check",
                "critical",
                0.9,
            )
        elif any(term in q for term in ["mang thai", "thai ky", "pregnant", "pregnancy", "breastfeeding", "cho con bu"]):
            category, intent, risk, confidence = (
                "pregnancy",
                "special_population",
                "critical",
                0.86,
            )
        elif any(term in q for term in ["tre em", "em be", "child", "children", "baby", "pediatric"]):
            category, intent, risk, confidence = (
                "pediatric",
                "special_population",
                "critical",
                0.84,
            )
        elif any(term in q for term in ["nguoi gia", "cao tuoi", "elderly", "geriatric", "senior"]):
            category, intent, risk, confidence = (
                "elderly",
                "special_population",
                "high",
                0.82,
            )
        elif has_known_drug or any(
            term in q
            for term in ["thuoc", "drug", "medication", "side effect", "tac dung phu", "dose", "lieu"]
        ):
            category, intent, risk, confidence = (
                "drug_safety",
                "drug_info",
                "high",
                0.82 if has_known_drug else 0.7,
            )
        elif scoped:
            best_category = self.safety_guard.best_category(query)
            category = (best_category or {}).get("id", "disease_knowledge")
            risk = (best_category or {}).get("risk_level", "medium")
            intent = "health_information"
            confidence = max(0.55, min(0.85, scope_score * 5))

        if category != "out_of_scope":
            requires_rag = True

        return QueryClassification(
            intent=intent,
            category=category,
            entities=entities,
            risk_level=risk if risk != "none" else "low",
            confidence=confidence,
            requires_rag=requires_rag,
            language=language,
        )

    def _extract_entities(self, normalized_query: str) -> list[str]:
        return extract_drug_entities(normalized_query, KNOWN_DRUGS)

    def _coerce_classification(
        self, data: dict, language: str
    ) -> QueryClassification:
        category = str(data.get("category") or "out_of_scope")
        if category == "general_health":
            category = "disease_knowledge"
        return QueryClassification(
            intent=str(data.get("intent") or category),
            category=category,
            entities=[str(item).lower() for item in data.get("entities", [])],
            risk_level=str(data.get("risk_level") or "medium"),
            confidence=float(data.get("confidence") or 0.0),
            requires_rag=bool(data.get("requires_rag", category != "out_of_scope")),
            language=language,
        )
