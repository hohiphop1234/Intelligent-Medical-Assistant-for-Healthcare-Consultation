from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from config import RAW_DATA_FALLBACK_DIRS
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
    "pregnancy & lactation": "pregnancy",
    "pregnancy": "pregnancy",
    "interactions": "drug_interaction",
    "contraindications": "drug_safety",
    "contraindication": "drug_safety",
    "safety": "drug_safety",
    "overdose": "overdose_triage",
    "pediatric": "pediatric",
    "edge_case": "drug_safety",
    "case_based": "drug_safety",
    "patient_query": "disease_knowledge",
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
        # Vietnamese website navigation/breadcrumb noise
        r"Sức khỏe\s+Quay lại\s+Sức khỏe\s+Quay lại",
        r"Trang chủ\s*[>\|»/]\s*",
        r"Quay lại\s+(?:Sức khỏe|Trang chủ|Danh mục)",
        r"Chia sẻ\s*(?:Facebook|Zalo|Twitter|Copy link).*?(?=\n|$)",
        r"(?:Đăng nhập|Đăng ký|Menu|Tìm kiếm)(?:\s*\|?\s*)+",
        r"(?:Bài viết liên quan|Xem thêm|Có thể bạn quan tâm).*?(?=\n|$)",
        r"(?:Nguồn|Tham khảo)\s*:?\s*https?://\S+",
        r"^\s*#\s*$",
    ]

    def fix_vietnamese_encoding(self, text: str) -> str:
        if ftfy is None:
            return text
        return ftfy.fix_text(text)

    def remove_noise(self, text: str) -> str:
        cleaned = text
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'</?think>', '', cleaned, flags=re.IGNORECASE)
        for pattern in self.NOISE_PATTERNS:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def clean_chunk_content(self, content: str) -> str:
        """Deep clean content: loại bỏ breadcrumbs, sửa encoding, chuẩn hóa."""
        content = self.fix_vietnamese_encoding(content)
        content = self.remove_noise(content)
        # Loại bỏ các dòng chỉ chứa navigation (< 3 từ, không chứa thuật ngữ y khoa)
        lines = content.split("\n")
        medical_terms = ["thuốc", "bệnh", "triệu chứng", "liều", "drug", "dose",
                         "medicine", "treatment", "điều trị", "tác dụng", "chống chỉ định"]
        cleaned_lines = [
            line for line in lines
            if len(line.split()) >= 3 or any(term in line.lower() for term in medical_terms)
        ]
        content = "\n".join(cleaned_lines)
        return self.normalize_whitespace(content)

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
        vi_files = set(
            f for f in (list(raw_path.glob("*vi*.jsonl")) + list(raw_path.glob("*hoangha*.jsonl")))
            if not f.name.startswith("clean_")
        )
        
        from tqdm import tqdm
        all_cleaned_rows = []
        for source_file in vi_files:
            rows = self._load_jsonl(source_file)
            for row in tqdm(rows, desc=f"🧹 Làm sạch & lọc rác ({source_file.name})", unit="chunk"):
                cleaned = self.clean_chunk(row)
                if cleaned is not None:
                    all_cleaned_rows.append(cleaned)
                    
        write_jsonl(output_path / "chunks_vi.jsonl", all_cleaned_rows)
        counts["vi"] = len(all_cleaned_rows)
        return counts

    def clean_chunk(self, row: dict[str, Any]) -> dict[str, Any] | None:
        content = str(row.get("content", ""))
        content = self.fix_vietnamese_encoding(content)
        content = self.remove_noise(content)
        content = self.normalize_whitespace(content)

        category = self._infer_category(row)
        risk_level = self._infer_risk(row, category)
        cleaned = {
            **row,
            "id": row.get("id") or self._make_id(row),
            "content": content,
            "language": "vi",
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
        if "pregnancy" in risk_categories or "lactation" in risk_categories:
            return "pregnancy"
        if "pediatric" in risk_categories:
            return "pediatric"
        return "disease_knowledge"

    def _infer_risk(self, row: dict[str, Any], category: str) -> str:
        risk_categories = " ".join(
            normalize_for_match(str(item))
            for item in row.get("supported_risk_categories", []) or []
        )
        if any(token in risk_categories for token in ["pregnancy", "pediatric", "overdose"]):
            return "critical"
        return CATEGORY_RISK_MAP.get(category, "medium")

    def _make_id(self, row: dict[str, Any]) -> str:
        seed = "|".join(
            [
                "vi",
                str(row.get("source", "")),
                str(row.get("url", "")),
                str(row.get("title", "")),
                str(row.get("content", ""))[:120],
            ]
        )
        return f"vi-{stable_hash(seed, 20)}"

    def _resolve_raw_dir(self, raw_dir: str) -> Path:
        candidates = self._candidate_dirs(raw_dir)
        valid_candidates = []
        for candidate in candidates:
            if any(candidate.glob("*vi*.jsonl")) or any(candidate.glob("*hoangha*.jsonl")):
                valid_candidates.append(candidate)
        if valid_candidates:
            requested = Path(raw_dir)
            if requested in valid_candidates:
                return requested
            return max(valid_candidates, key=self._latest_data_mtime)
        raise FileNotFoundError(
            "Could not find Vietnamese chunks in "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    def _candidate_dirs(self, raw_dir: str) -> list[Path]:
        candidates: list[Path] = []
        seen: set[Path] = set()

        def add(candidate: str | Path) -> None:
            path = Path(candidate).expanduser()
            variants = [path]
            if path.name != "rag_processed":
                variants.extend([path / "rag_processed", path / "processed", path / "raw"])
            for variant in variants:
                normalized = variant.resolve() if variant.exists() else variant
                if normalized not in seen:
                    seen.add(normalized)
                    candidates.append(variant)

        add(raw_dir)
        for fallback in RAW_DATA_FALLBACK_DIRS:
            add(fallback)
        return candidates

    def _latest_data_mtime(self, path: Path) -> float:
        files = list(path.glob("*vi*.jsonl")) + list(path.glob("*hoangha*.jsonl"))
        return max((f.stat().st_mtime for f in files if f.exists()), default=0.0)

    def _first_existing(self, raw_path: Path, filenames: list[str]) -> Path:
        for filename in filenames:
            candidate = raw_path / filename
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not find any of {', '.join(filenames)} in {raw_path}"
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
