from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from config import RAW_DATA_FALLBACK_DIR
from src.utils import ensure_dir, normalize_for_match, stable_hash, write_jsonl

try:
    import ftfy
except ImportError:  # pragma: no cover - optional dependency
    ftfy = None


TOPIC_CATEGORY_MAP = {
    "drug safety": "drug_safety",
    "overdose & triage": "overdose_triage",
    "disease knowledge": "disease_knowledge",
    "pediatric & special populations": "pediatric",
}

CATEGORY_RISK_MAP = {
    "drug_safety": "high",
    "drug_interaction": "critical",
    "overdose_triage": "critical",
    "disease_knowledge": "medium",
    "pregnancy": "critical",
    "pediatric": "critical",
    "elderly": "high",
}


class DataCleaner:
    """Clean crawled JSONL chunks into RAG-ready records."""

    NOISE_PATTERNS = [
        r"An official website of the United States government.*?Share sensitive information only on official, secure websites\.",
        r"Official websites use \.gov.*?secure websites\.",
        r"Skip to main content",
        r"U\.S\. National Library of Medicine",
        r"National Library of Medicine",
        r"MedlinePlus.*?Trusted Health Information",
        r"You Are Here:.*?(?=#|\n|$)",
        r"URL of this page:.*?(?=#|\n|$)",
        r"Show Search Search MedlinePlus GO",
        r"Page last updated:.*",
        r"Copyright.*",
        r"Menu\s+\*",
        r"\[\s*\]\([^)]*\)",
    ]

    def fix_vietnamese_encoding(self, text: str) -> str:
        if ftfy is None:
            return text
        return ftfy.fix_text(text)

    def remove_noise(self, text: str) -> str:
        cleaned = text
        for pattern in self.NOISE_PATTERNS:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def validate_chunk(self, chunk: dict[str, Any]) -> bool:
        content = chunk.get("content", "")
        if not content:
            return False
        words = content.split()
        if len(words) < 50:
            return False
        special_ratio = sum(
            1 for char in content if not char.isalnum() and not char.isspace()
        ) / max(len(content), 1)
        if special_ratio > 0.3:
            return False
        return True

    def quality_score(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        special_ratio = sum(
            1 for char in text if not char.isalnum() and not char.isspace()
        ) / max(len(text), 1)
        length_score = min(len(words) / 150, 1.0)
        noise_penalty = min(special_ratio / 0.3, 1.0)
        return round(max(0.0, (0.7 * length_score) + (0.3 * (1 - noise_penalty))), 3)

    def process_dataset(self, raw_dir: str, output_dir: str) -> dict[str, int]:
        raw_path = self._resolve_raw_dir(raw_dir)
        output_path = ensure_dir(output_dir)

        counts = {}
        for language, filename, output_name in [
            ("en", "rag_chunks.jsonl", "chunks_en.jsonl"),
            ("vi", "rag_chunks_vi.jsonl", "chunks_vi.jsonl"),
        ]:
            source_file = raw_path / filename
            rows = self._load_jsonl(source_file)
            cleaned_rows = [
                cleaned
                for row in rows
                if (cleaned := self.clean_chunk(row, language)) is not None
            ]
            write_jsonl(output_path / output_name, cleaned_rows)
            counts[language] = len(cleaned_rows)
        return counts

    def clean_chunk(self, row: dict[str, Any], language: str) -> dict[str, Any] | None:
        content = str(row.get("content", ""))
        if language == "vi":
            content = self.fix_vietnamese_encoding(content)
        content = self.remove_noise(content)
        content = self.normalize_whitespace(content)

        category = self._infer_category(row)
        risk_level = self._infer_risk(row, category)
        cleaned = {
            **row,
            "id": row.get("id") or self._make_id(row, language),
            "content": content,
            "language": language,
            "word_count": len(content.split()),
            "quality_score": self.quality_score(content),
            "category": category,
            "risk_level": risk_level,
        }

        if not self.validate_chunk(cleaned):
            return None
        return cleaned

    def _infer_category(self, row: dict[str, Any]) -> str:
        topic = normalize_for_match(str(row.get("topic_group", "")))
        for key, category in TOPIC_CATEGORY_MAP.items():
            if key in topic:
                return category

        risk_categories = [
            normalize_for_match(str(item))
            for item in row.get("supported_risk_categories", []) or []
        ]
        if "overdose" in risk_categories:
            return "overdose_triage"
        return "disease_knowledge"

    def _infer_risk(self, row: dict[str, Any], category: str) -> str:
        risk_categories = " ".join(
            normalize_for_match(str(item))
            for item in row.get("supported_risk_categories", []) or []
        )
        if any(token in risk_categories for token in ["pregnancy", "pediatric", "overdose"]):
            return "critical"
        return CATEGORY_RISK_MAP.get(category, "medium")

    def _make_id(self, row: dict[str, Any], language: str) -> str:
        seed = "|".join(
            [
                language,
                str(row.get("source", "")),
                str(row.get("url", "")),
                str(row.get("title", "")),
                str(row.get("content", ""))[:120],
            ]
        )
        return f"{language}-{stable_hash(seed, 20)}"

    def _resolve_raw_dir(self, raw_dir: str) -> Path:
        candidates = [Path(raw_dir), Path(RAW_DATA_FALLBACK_DIR)]
        for candidate in candidates:
            if (candidate / "rag_chunks.jsonl").exists() and (
                candidate / "rag_chunks_vi.jsonl"
            ).exists():
                return candidate
        raise FileNotFoundError(
            "Could not find rag_chunks.jsonl and rag_chunks_vi.jsonl in "
            f"{raw_dir} or {RAW_DATA_FALLBACK_DIR}"
        )

    def _load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        import json

        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
        return rows
