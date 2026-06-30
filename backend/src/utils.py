from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


DRUG_ALIASES = {
    "panadol": "acetaminophen",
    "tylenol": "acetaminophen",
    "paracetamol": "acetaminophen",
}

DRUG_EXPANSIONS = {
    "acetaminophen": ["acetaminophen", "paracetamol"],
}


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def strip_accents(text: str) -> str:
    text = text.replace('đ', 'd').replace('Đ', 'D')
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_for_match(text: str) -> str:
    text = strip_accents(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_for_match(text)
    return [token for token in normalized.split() if len(token) > 1]


def canonicalize_drug_name(name: str) -> str:
    normalized = normalize_for_match(name)
    return DRUG_ALIASES.get(normalized, normalized)


def extract_drug_entities(text: str, known_drugs: Iterable[str]) -> list[str]:
    normalized = normalize_for_match(text)
    entities = set()
    for drug in known_drugs:
        normalized_drug = normalize_for_match(drug)
        if normalized_drug in normalized:
            entities.add(canonicalize_drug_name(normalized_drug))
    for alias, canonical in DRUG_ALIASES.items():
        if alias in normalized:
            entities.add(canonical)
    return sorted(entities)


def expand_query_with_drug_aliases(query: str) -> str:
    entities = extract_drug_entities(
        query,
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
        ],
    )
    extra_terms = []
    for entity in entities:
        extra_terms.extend(DRUG_EXPANSIONS.get(entity, [entity]))
    extra_terms = [term for term in dict.fromkeys(extra_terms) if term not in normalize_for_match(query)]
    if not extra_terms:
        return query
    return f"{query} {' '.join(extra_terms)}"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def chunk_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        part = words[start : start + chunk_size]
        if part:
            chunks.append(" ".join(part))
        if start + chunk_size >= len(words):
            break
    return chunks


def safe_json_loads(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
