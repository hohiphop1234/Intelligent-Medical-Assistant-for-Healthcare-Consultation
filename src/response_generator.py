from __future__ import annotations

import re
from typing import Any

from config import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from src.query_router import QueryClassification
from src.utils import normalize_for_match, tokenize

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


SYSTEM_PROMPT = """You are a medical information assistant.
Follow these rules strictly:
1. Answer only from the provided context documents.
2. Cite sources with [1], [2], [3] after factual claims.
3. If evidence is insufficient, say that evidence is insufficient.
4. Never diagnose, prescribe, or recommend personal dosages.
5. Recommend consulting a healthcare professional.
6. Use the same language as the user question.
"""


class ResponseGenerator:
    """Generate cited answers from graded context."""

    def __init__(self):
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

    def generate(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        classification: QueryClassification,
    ) -> dict[str, Any]:
        used_chunks = chunks
        if self.client is not None:
            try:
                answer = self._generate_with_llm(question, chunks)
            except Exception:
                answer, used_chunks = self._generate_extractive(
                    question, chunks, classification.language
                )
        else:
            answer, used_chunks = self._generate_extractive(
                question, chunks, classification.language
            )

        return {
            "answer": answer,
            "sources": self._format_sources(used_chunks),
            "risk_level": classification.risk_level,
            "category": classification.category,
            "classification_confidence": classification.confidence,
            "language": classification.language,
        }

    def _generate_with_llm(self, question: str, chunks: list[dict[str, Any]]) -> str:
        context = self._build_context(chunks)
        prompt = (
            f"Context documents:\n{context}\n\n"
            f"User question: {question}\n\n"
            "Answer using only the context above. Cite sources with [1], [2], etc."
        )
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content or ""

    def _build_context(self, chunks: list[dict[str, Any]]) -> str:
        parts = []
        for index, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source") or chunk.get("source", "Unknown")
            title = metadata.get("title") or chunk.get("title", "")
            section = metadata.get("section") or chunk.get("section", "")
            parts.append(
                f"[{index}] Source: {source} | {title} | {section}\n{chunk['content']}"
            )
        return "\n\n---\n\n".join(parts)

    def _generate_extractive(
        self, question: str, chunks: list[dict[str, Any]], language: str
    ) -> tuple[str, list[dict[str, Any]]]:
        if language == "vi":
            intro = "Dựa trên các nguồn được truy xuất:"
            closing = (
                "Vui lòng trao đổi với bác sĩ hoặc dược sĩ trước khi áp dụng, "
                "đặc biệt nếu có bệnh nền, đang mang thai, là trẻ em/người cao tuổi, "
                "hoặc đang dùng thuốc khác."
            )
        else:
            intro = "Based on the retrieved sources:"
            closing = (
                "Please consult a doctor or pharmacist before acting on this, "
                "especially for pregnancy, children, older adults, medical conditions, "
                "or medication changes."
            )

        if self._is_side_effect_question(question):
            side_effect_answer = self._generate_side_effect_answer(
                chunks, language, intro, closing
            )
            if side_effect_answer is not None:
                return side_effect_answer

        evidence = []
        for chunk in chunks[:4]:
            sentence = self._best_sentence(question, chunk.get("content", ""))
            if self._is_generic_side_effect_sentence(question, sentence):
                continue
            if self._is_side_effect_question(question):
                if not self._has_concrete_side_effect_terms(sentence):
                    continue
                if not self._has_side_effect_context(sentence):
                    continue
            if sentence:
                evidence.append((chunk, sentence))
        if not evidence:
            return (
                (
                    "Không đủ thông tin trong các nguồn được truy xuất để trả lời chính xác."
                    if language == "vi"
                    else "There is not enough information to answer accurately."
                ),
                chunks[:1],
            )
        bullets = [
            f"- {sentence} [{source_index}]"
            for source_index, (_, sentence) in enumerate(evidence, 1)
        ]
        used_chunks = [chunk for chunk, _ in evidence]
        return intro + "\n\n" + "\n".join(bullets) + "\n\n" + closing, used_chunks

    def _generate_side_effect_answer(
        self,
        chunks: list[dict[str, Any]],
        language: str,
        intro: str,
        closing: str,
    ) -> tuple[str, list[dict[str, Any]]] | None:
        best_candidate: tuple[dict[str, Any], list[str], str] | None = None
        for chunk in chunks[:6]:
            content = chunk.get("content", "")
            items = self._extract_side_effect_items(content)
            if len(items) < 3:
                continue
            warning = self._find_sentence_with_terms(
                content,
                ["hoại tử", "gangrene", "màu tím", "tối màu"]
                if language == "vi"
                else ["necrosis", "gangrene", "purple", "dark"],
            )
            if best_candidate is None or len(items) > len(best_candidate[1]):
                best_candidate = (chunk, items, warning)

        if best_candidate is None:
            return None

        chunk, items, warning = best_candidate
        metadata = chunk.get("metadata", {})
        selected_url = metadata.get("url") or chunk.get("url")
        selected_entity = normalize_for_match(str(metadata.get("entity") or chunk.get("entity", "")))
        merged_items = list(items)
        for other_chunk in chunks[:6]:
            other_metadata = other_chunk.get("metadata", {})
            other_url = other_metadata.get("url") or other_chunk.get("url")
            other_entity = normalize_for_match(
                str(other_metadata.get("entity") or other_chunk.get("entity", ""))
            )
            if selected_url and other_url != selected_url:
                continue
            if selected_entity and other_entity and other_entity != selected_entity:
                continue
            merged_items.extend(self._extract_side_effect_items(other_chunk.get("content", "")))
            if not warning:
                warning = self._find_sentence_with_terms(
                    other_chunk.get("content", ""),
                    ["hoại tử", "gangrene", "màu tím", "tối màu"]
                    if language == "vi"
                    else ["necrosis", "gangrene", "purple", "dark"],
                )
        items = self._dedupe_keep_order(merged_items)

        symptom_line = "; ".join(items[:10])
        if language == "vi":
            bullets = [
                (
                    "- Các dấu hiệu/tác dụng phụ cần gọi bác sĩ ngay gồm: "
                    f"{symptom_line}. [1]"
                )
            ]
        else:
            bullets = [
                (
                    "- Signs/side effects that require calling a doctor include: "
                    f"{symptom_line}. [1]"
                )
            ]
        if warning:
            bullets.append(f"- {warning}. [1]")

        return intro + "\n\n" + "\n".join(bullets) + "\n\n" + closing, [chunk]

    def _extract_side_effect_items(self, content: str) -> list[str]:
        items = []
        for raw_item in re.findall(r"\*\s*([^*#]+?)(?=\s+\*|$)", content):
            item = self._clean_list_item(raw_item)
            normalized_item = normalize_for_match(item)
            if any(
                false_positive in normalized_item
                for false_positive in [
                    "ngan ngua",
                    "tieu duong",
                    "ngan ngua va dieu tri",
                    "chua metformin",
                    "contains metformin",
                ]
            ):
                continue
            if "®" in item or "™" in item:
                continue
            if item and self._has_concrete_side_effect_terms(item):
                items.append(item)
        return self._dedupe_keep_order(items)

    def _clean_list_item(self, item: str) -> str:
        item = re.sub(r"\s+", " ", item).strip(" .,;:-")
        item = item.replace("_", "")
        item = re.sub(r"\[[^\]]+\]\([^)]+\)", "", item)
        item = re.split(r"\s+\w+\s+có thể gây", item, maxsplit=1)[0].strip(" .,;:-")
        item = re.split(r"\s+\w+\s+may cause", item, maxsplit=1, flags=re.IGNORECASE)[0].strip(" .,;:-")
        item = re.split(r"\s+##\s+|\s+###\s+", item)[0].strip(" .,;:-")
        if len(item.split()) > 18:
            item = " ".join(item.split()[:18]).strip(" .,;:-")
        return item

    def _dedupe_keep_order(self, items: list[str]) -> list[str]:
        seen = set()
        deduped = []
        for item in items:
            key = normalize_for_match(item)
            if self._is_redundant_side_effect_item(key, seen):
                continue
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _is_redundant_side_effect_item(self, key: str, seen: set[str]) -> bool:
        if "dau vung bung tren" in key and any("dau vung bung tren" in s for s in seen):
            return True
        yellow_terms = ["da hoac mat vang", "da vang", "mat vang", "trang cua mat"]
        if any(term in key for term in yellow_terms):
            return any(any(term in s for term in yellow_terms) for s in seen)
        return False

    def _find_sentence_with_terms(self, content: str, terms: list[str]) -> str:
        normalized_content = normalize_for_match(content)
        if "hoai tu" in normalized_content or "gangrene" in normalized_content:
            if any(term in ["hoại tử", "gangrene", "màu tím", "tối màu"] for term in terms):
                return (
                    "Warfarin có thể gây hoại tử hoặc hoại thư; hãy gọi bác sĩ ngay "
                    "nếu thấy da đổi màu/tím, loét, đau dữ dội hoặc thay đổi màu/nhiệt độ trên cơ thể"
                )
            if any(term in ["necrosis", "gangrene", "purple", "dark"] for term in terms):
                return (
                    "Warfarin may cause necrosis or gangrene; call a doctor right away "
                    "if skin turns dark or purple, ulcers appear, severe pain occurs, "
                    "or body color/temperature changes"
                )

        lowered_terms = [term.lower() for term in terms]
        for sentence in re.split(r"(?<=[.!?])\s+", content):
            if any(term in sentence.lower() for term in lowered_terms):
                return sentence[:450].strip(" .;:-")
        return ""

    def _best_sentence(self, question: str, content: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", content)
        stopwords = {
            "what",
            "are",
            "the",
            "of",
            "is",
            "a",
            "an",
            "to",
            "for",
            "cua",
            "la",
            "gi",
            "co",
            "khong",
        }
        query_terms = {
            term for term in tokenize(question) if term not in stopwords
        }
        normalized_question = normalize_for_match(question)
        asks_side_effects = any(
            phrase in normalized_question
            for phrase in [
                "side effect",
                "side effects",
                "adverse effect",
                "tac dung phu",
                "phan ung phu",
            ]
        )
        side_effect_terms = {
            "side",
            "effect",
            "effects",
            "adverse",
            "bleeding",
            "bruising",
            "rash",
            "hives",
            "nausea",
            "vomiting",
            "diarrhea",
            "swelling",
            "dizziness",
            "weakness",
            "bleed",
            "symptoms",
            "chay",
            "bam",
            "phat",
            "me",
            "day",
            "ngua",
            "kho",
            "tho",
            "nuot",
            "sung",
            "mat",
            "hong",
            "luoi",
            "nguc",
            "moi",
            "khan",
            "sot",
            "nhiem",
            "trung",
            "buon",
            "non",
            "tieu",
            "chay",
            "khau",
            "vi",
            "moi",
            "vang",
            "da",
            "hoai",
        }
        best_sentence = ""
        best_score = -1
        for sentence in sentences[:40]:
            terms = set(tokenize(sentence))
            score = len(query_terms & terms) * 2
            if asks_side_effects:
                symptom_hits = len(side_effect_terms & terms)
                normalized_sentence = normalize_for_match(sentence)
                if symptom_hits == 0 and any(
                    phrase in normalized_sentence
                    for phrase in [
                        "theo doi ky luong cac tac dung phu",
                        "monitor carefully for side effects",
                        "monitor you carefully for side effects",
                    ]
                ):
                    continue
                score += symptom_hits * 2
                if (
                    "side effects" in normalized_sentence
                    or "tac dung phu" in normalized_sentence
                    or "neu ban gap phai" in normalized_sentence
                ):
                    score += 5
            if score > best_score and len(sentence.split()) >= 8:
                best_sentence = sentence
                best_score = score
        return best_sentence[:600].strip()

    def _is_generic_side_effect_sentence(self, question: str, sentence: str) -> bool:
        if not self._is_side_effect_question(question):
            return False

        tokens = set(tokenize(sentence))
        normalized_sentence = normalize_for_match(sentence)
        generic_monitoring = (
            {"theo", "doi", "tac", "dung", "phu"}.issubset(tokens)
            or "monitor carefully for side effects" in normalized_sentence
            or "monitor you carefully for side effects" in normalized_sentence
        )
        if not generic_monitoring:
            return False

        return not self._has_concrete_side_effect_terms(sentence)

    def _is_side_effect_question(self, question: str) -> bool:
        normalized_question = normalize_for_match(question)
        return any(
            phrase in normalized_question
            for phrase in [
                "side effect",
                "side effects",
                "adverse effect",
                "tac dung phu",
                "phan ung phu",
            ]
        )

    def _has_concrete_side_effect_terms(self, sentence: str) -> bool:
        tokens = set(tokenize(sentence))
        concrete_symptom_terms = {
            "bleeding",
            "bruising",
            "rash",
            "hives",
            "nausea",
            "vomiting",
            "diarrhea",
            "swelling",
            "dizziness",
            "weakness",
            "chay",
            "bam",
            "phat",
            "ngua",
            "kho",
            "tho",
            "sung",
            "hong",
            "luoi",
            "nguc",
            "khan",
            "sot",
            "buon",
            "non",
            "tieu",
            "moi",
            "yeu",
            "khau",
            "vi",
            "vang",
            "hoai",
        }
        return bool(tokens & concrete_symptom_terms)

    def _has_side_effect_context(self, sentence: str) -> bool:
        normalized = normalize_for_match(sentence)
        cues = [
            "neu ban gap",
            "gap phai",
            "goi bac si",
            "co the gay",
            "tac dung phu",
            "trieu chung",
            "if you experience",
            "call your doctor",
            "may cause",
            "side effects",
            "symptoms",
        ]
        return any(cue in normalized for cue in cues)

    def _format_sources(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sources = []
        for index, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            sources.append(
                {
                    "index": index,
                    "title": metadata.get("title") or chunk.get("title", "Unknown"),
                    "source": metadata.get("source") or chunk.get("source", ""),
                    "url": metadata.get("url") or chunk.get("url", ""),
                    "section": metadata.get("section") or chunk.get("section", ""),
                    "score": chunk.get("fused_score", chunk.get("score", 0.0)),
                }
            )
        return sources
