from __future__ import annotations

import re
from typing import Any

from src.utils import normalize_for_match


PREGNANCY_QUERY_TERMS = (
    "mang thai",
    "co thai",
    "thai ky",
    "ba bau",
    "pregnan",
    "fetal",
    "fetus",
    "thai nhi",
)

LACTATION_QUERY_TERMS = (
    "cho con bu",
    "dang cho bu",
    "sua me",
    "nuoi con bang sua me",
    "breastfeed",
    "breastfeeding",
    "breast milk",
    "lactation",
    "lactating",
    "postpartum",
    "sau sinh",
)

MEDICATION_QUERY_TERMS = (
    "thuoc",
    "uong",
    "dung",
    "medicine",
    "medication",
    "drug",
    "take",
    "use",
)

AVOIDANCE_QUERY_TERMS = (
    "khong nen",
    "khong duoc",
    "khong dung",
    "khong nen uong",
    "can tranh",
    "tranh",
    "chong chi dinh",
    "avoid",
    "not take",
    "do not take",
    "should not",
    "contraindicat",
    "unsafe",
)

PREGNANCY_CONTEXT_TERMS = (
    "mang thai",
    "co thai",
    "thai ky",
    "ba bau",
    "pregnan",
    "fetal",
    "fetus",
    "thai nhi",
    "unborn",
    "birth defect",
    "di tat",
    "tu cung",
    "gestational",
    "tuan thu 20",
    "20 weeks",
)

PREGNANCY_STRONG_RISK_TERMS = (
    "co the gay hai cho thai nhi",
    "gay hai cho thai nhi",
    "harm the fetus",
    "harm an unborn baby",
    "can harm an unborn baby",
    "may harm an unborn baby",
    "birth defect",
    "di tat",
    "gay di tat",
    "fetal harm",
    "tuan thu 20",
    "20 weeks",
)

GENERIC_AVOIDANCE_CONTENT_TERMS = (
    "khong nen dung",
    "khong nen uong",
    "khong duoc dung",
    "khong duoc uong",
    "khong dung",
    "chong chi dinh",
    "do not take",
    "should not take",
    "not take",
    "avoid taking",
    "avoid using",
    "avoid",
    "contraindicat",
)

LACTATION_CONTEXT_TERMS = (
    "cho con bu",
    "dang cho bu",
    "sua me",
    "tiet sua",
    "tao sua",
    "tre bu me",
    "breastfeed",
    "breastfeeding",
    "breastfed",
    "breast milk",
    "breastmilk",
    "lactation",
    "lactating",
    "milk level",
    "milk concentration",
    "postpartum",
    "sau sinh",
)

LACTATION_ONLY_SECTIONS = (
    "summary of use during lactation",
    "drug levels",
    "effects in breastfed infants",
    "effects on lactation",
    "breastfeeding",
    "references",
    "recent activity",
)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _count_terms(text: str, terms: tuple[str, ...]) -> int:
    return sum(text.count(term) for term in terms)


def is_pregnancy_query(query: str) -> bool:
    normalized = normalize_for_match(query)
    return _contains_any(normalized, PREGNANCY_QUERY_TERMS)


def is_lactation_query(query: str) -> bool:
    normalized = normalize_for_match(query)
    return _contains_any(normalized, LACTATION_QUERY_TERMS)


def is_medication_query(query: str) -> bool:
    normalized = normalize_for_match(query)
    return _contains_any(normalized, MEDICATION_QUERY_TERMS)


def asks_drug_avoidance(query: str) -> bool:
    normalized = normalize_for_match(query)
    return _contains_any(normalized, AVOIDANCE_QUERY_TERMS)


def has_pregnancy_context(content: str) -> bool:
    normalized = normalize_for_match(content)
    return _contains_any(normalized, PREGNANCY_CONTEXT_TERMS)


def has_pregnancy_risk_or_avoidance(content: str) -> bool:
    normalized = normalize_for_match(content)
    if _contains_any(normalized, PREGNANCY_STRONG_RISK_TERMS):
        return True

    for sentence in re.split(r"(?<=[.!?])\s+|\s+\*\s+", content):
        normalized_sentence = normalize_for_match(sentence)
        if _contains_any(
            normalized_sentence, PREGNANCY_CONTEXT_TERMS
        ) and _contains_any(normalized_sentence, GENERIC_AVOIDANCE_CONTENT_TERMS):
            return True
    return False


def has_lactation_context(content: str) -> bool:
    normalized = normalize_for_match(content)
    return _contains_any(normalized, LACTATION_CONTEXT_TERMS)


def should_route_pregnancy_to_drug_safety(query: str) -> bool:
    return (
        is_pregnancy_query(query)
        and not is_lactation_query(query)
        and is_medication_query(query)
    )


def is_lactation_only_evidence(content: str, metadata: dict[str, Any] | None = None) -> bool:
    metadata = metadata or {}
    normalized_content = normalize_for_match(content)
    section = normalize_for_match(str(metadata.get("section", "")))
    source_title = normalize_for_match(
        " ".join(
            str(metadata.get(key, ""))
            for key in ("source", "title", "document_type", "topic_group")
        )
    )

    if has_pregnancy_risk_or_avoidance(normalized_content):
        return False

    if "lactmed" in source_title and _contains_any(section, LACTATION_ONLY_SECTIONS):
        return True

    lactation_hits = _count_terms(normalized_content, LACTATION_CONTEXT_TERMS)
    pregnancy_hits = _count_terms(normalized_content, PREGNANCY_CONTEXT_TERMS)
    return lactation_hits > pregnancy_hits


def pregnancy_hard_reject(
    question: str, content: str, metadata: dict[str, Any] | None = None
) -> bool:
    if not is_pregnancy_query(question) or is_lactation_query(question):
        return False

    if is_lactation_only_evidence(content, metadata):
        return True

    if not has_pregnancy_context(content):
        return True

    if asks_drug_avoidance(question) and not has_pregnancy_risk_or_avoidance(content):
        return True

    return False


def pregnancy_relevance_bonus(
    question: str, content: str, metadata: dict[str, Any] | None = None
) -> float:
    if not is_pregnancy_query(question) or is_lactation_query(question):
        return 0.0

    if pregnancy_hard_reject(question, content, metadata):
        return -0.12

    bonus = 0.0
    if has_pregnancy_context(content):
        bonus += 0.18
    if has_pregnancy_risk_or_avoidance(content):
        bonus += 0.28
    if asks_drug_avoidance(question) and has_pregnancy_risk_or_avoidance(content):
        bonus += 0.12

    metadata = metadata or {}
    if str(metadata.get("category", "")) == "drug_safety" and is_medication_query(question):
        bonus += 0.05
    return bonus
