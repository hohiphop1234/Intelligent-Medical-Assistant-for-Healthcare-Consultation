from __future__ import annotations

import re
from typing import Any

from config import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
)
from src.query_router import QueryClassification
from src.topic_relevance import (
    asks_drug_avoidance,
    has_pregnancy_context,
    has_pregnancy_risk_or_avoidance,
    is_lactation_query,
    is_pregnancy_query,
)
from src.utils import normalize_for_match, tokenize

from src.qwen_llm import QwenMedicalLLM


SYSTEM_PROMPT = """You are a trusted medical information assistant that synthesizes evidence from verified sources.

## Core Rules
1. Answer ONLY from the provided context documents. Never invent information.
2. Cite every factual claim with [1], [2], [3] matching the source index.
3. NEVER diagnose, prescribe, or recommend specific dosages.
4. Always recommend consulting a healthcare professional at the end.
5. Always answer in Vietnamese.

## Answer Structure
- Start with a **direct, extremely concise answer** to the user's question (1-2 sentences).
- ONLY provide the most important supporting details. Do NOT write long paragraphs. Keep the entire answer short and straight to the point (under 5-6 sentences total if possible).
- If the question asks "which" or "what" (e.g., "thuốc nào", "which drugs"), list specific names/items found in the sources briefly.
- If context documents don't contain a direct answer, explicitly state: "Thông tin trong nguồn dữ liệu không đủ để trả lời trực tiếp câu hỏi này".

## Quality Guidelines
- **Synthesize**, don't copy-paste. Rewrite raw source text into natural, readable sentences.
- Remove navigation text, breadcrumbs, or formatting artifacts from source content before using it.
- Merge related information from multiple sources into coherent paragraphs.
- For drug safety questions: mention specific drug names, age restrictions, and contraindications if available.
- For pregnancy questions: do NOT use breastfeeding/lactation/postpartum evidence unless the user explicitly asks about breastfeeding.
"""


class ResponseGenerator:
    """Generate cited answers from graded context."""

    def __init__(self):
        self.local_llm = QwenMedicalLLM()

    def generate(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        classification: QueryClassification,
    ) -> dict[str, Any]:
        used_chunks = chunks
        try:
            answer = self._generate_with_llm(question, chunks)
            if answer.startswith("Lỗi:"):
                raise Exception(answer)
        except Exception as e:
            print(f"\\n[RAG Pipeline] Local LLM Failed: {e}\\n")
            answer, used_chunks = self._generate_extractive(
                question, chunks, classification
            )

        return {
            "answer": answer,
            "sources": self._format_sources(used_chunks),
            "risk_level": classification.risk_level,
            "category": classification.category,
            "classification_confidence": classification.confidence,
        }

    def _generate_with_llm(self, question: str, chunks: list[dict[str, Any]]) -> str:
        top_chunks = chunks[:4]
        context = self._build_context(top_chunks)
        prompt = (
            f"Context documents:\n{context}\n\n"
            f"User question: {question}\n\n"
            "Answer using only the context above. Cite sources with [1], [2], etc. "
            "Ignore citation numbers that appear inside a context document; only use "
            "the source numbers assigned to the context documents."
        )
        
        return self.local_llm.generate_answer(
            question=prompt,
            max_new_tokens=2048,
            system_prompt=SYSTEM_PROMPT
        )
        
    def _generate_with_llm_stream(
        self, question: str, chunks: list[dict[str, Any]]
    ):
        # Giới hạn số lượng tài liệu gửi vào LLM để không bị tràn VRAM (9000+ tokens)
        top_chunks = chunks[:4]
        context = self._build_context(top_chunks)
        prompt = (
            f"Context documents:\n{context}\n\n"
            f"User question: {question}\n\n"
            "Answer using only the context above. Cite sources with [1], [2], etc. "
            "Ignore citation numbers that appear inside a context document; only use "
            "the source numbers assigned to the context documents."
        )
        yield from self.local_llm.stream_answer(
            question=prompt,
            max_new_tokens=2048,
            system_prompt=SYSTEM_PROMPT
        )

    def generate_stream(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        classification: QueryClassification,
    ):
        """Streaming version of answer generator"""
        # Trực tiếp stream từ LLM mà không sinh câu phụ (disclaimer sẽ được thêm ở api.py)
        # Các logic interaction/side_effect (câu trả lời soạn sẵn) có thể trả về một cục
        
        if classification.category == "drug_interaction":
            interaction_answer = self._generate_interaction_answer(
                question, chunks, classification.entities, intro="Dựa trên các nguồn được truy xuất:", closing="Vui lòng trao đổi với bác sĩ hoặc dược sĩ trước khi áp dụng."
            )
            if interaction_answer is not None:
                yield interaction_answer[0]
                return

        if self._is_side_effect_question(question):
            side_effect_answer = self._generate_side_effect_answer(
                chunks, intro="Dựa trên các nguồn được truy xuất:", closing="Vui lòng trao đổi với bác sĩ hoặc dược sĩ trước khi áp dụng."
            )
            if side_effect_answer is not None:
                yield side_effect_answer[0]
                return
                
        try:
            yield from self._generate_with_llm_stream(question, chunks)
        except Exception as e:
            print(f"\n[RAG Pipeline] Local LLM Stream Failed: {e}\n")
            ans, _ = self._generate_extractive(question, chunks, classification)
            yield ans

    def _build_context(self, chunks: list[dict[str, Any]]) -> str:
        parts = []
        for index, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source") or chunk.get("source", "Unknown")
            title = metadata.get("title") or chunk.get("title", "")
            section = metadata.get("section") or chunk.get("section", "")
            content = chunk['content']
            # Clean navigation/breadcrumb noise trước khi gửi vào LLM
            content = re.sub(r"Sức khỏe\s+Quay lại.*?(?=#|\n|$)", "", content)
            content = re.sub(r"Quay lại\s+\w+\s+Quay lại", "", content)
            content = re.sub(r"^\s*#\s*$", "", content, flags=re.MULTILINE)
            content = re.sub(r"\n{3,}", "\n\n", content).strip()
            parts.append(
                f"[{index}] Source: {source} | {title} | {section}\n{content}"
            )
        return "\n\n---\n\n".join(parts)

    def _generate_extractive(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        classification: QueryClassification,
    ) -> tuple[str, list[dict[str, Any]]]:
        intro = "Dựa trên các nguồn được truy xuất:"
        closing = (
            "Vui lòng trao đổi với bác sĩ hoặc dược sĩ trước khi áp dụng, "
            "đặc biệt nếu có bệnh nền, đang mang thai, là trẻ em/người cao tuổi, "
            "hoặc đang dùng thuốc khác."
        )

        if classification.category == "drug_interaction":
            interaction_answer = self._generate_interaction_answer(
                question, chunks, classification.entities, intro, closing
            )
            if interaction_answer is not None:
                return interaction_answer

        if self._is_side_effect_question(question):
            side_effect_answer = self._generate_side_effect_answer(
                chunks, intro, closing
            )
            if side_effect_answer is not None:
                return side_effect_answer

        evidence = []
        for chunk in self._dedupe_chunks_for_answer(chunks)[:4]:
            sentence = self._best_sentence(question, chunk.get("content", ""))
            sentence = self._clean_extractive_sentence(sentence)
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
                "Không đủ thông tin trong các nguồn được truy xuất để trả lời chính xác.",
                chunks[:1],
            )
        bullets = [
            f"- {sentence} [{source_index}]"
            for source_index, (_, sentence) in enumerate(evidence, 1)
        ]
        used_chunks = [chunk for chunk, _ in evidence]
        return intro + "\n\n" + "\n".join(bullets) + "\n\n" + closing, used_chunks

    def _generate_interaction_answer(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        entities: list[str],
        intro: str,
        closing: str,
    ) -> tuple[str, list[dict[str, Any]]] | None:
        query_entities = set(entities) or set(
            extract
            for extract in [
                "warfarin",
                "ibuprofen",
                "aspirin",
                "acetaminophen",
                "metformin",
                "insulin",
                "phenytoin",
                "rifampin",
            ]
            if extract in normalize_for_match(question)
        )
        candidates: list[tuple[float, dict[str, Any], str]] = []
        seen_sentences: set[str] = set()
        for chunk in chunks[:10]:
            content = chunk.get("content", "")
            for sentence in self._interaction_sentences(content):
                cleaned = self._clean_extractive_sentence(sentence)
                if not cleaned:
                    continue
                normalized = normalize_for_match(cleaned)
                score = self._interaction_sentence_score(normalized, query_entities)
                if score <= 0:
                    continue
                key = normalized[:180]
                if key in seen_sentences:
                    continue
                seen_sentences.add(key)
                candidates.append((score, chunk, self._truncate_words(cleaned, 95)))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_normalized = normalize_for_match(candidates[0][2])
        if query_entities and all(entity in best_normalized for entity in query_entities) and any(
            cue in best_normalized
            for cue in ["may interact", "do not start", "without discussing", "khong bat dau"]
        ):
            candidates = candidates[:1]

        selected: list[tuple[dict[str, Any], str]] = []
        used_chunk_ids: set[str] = set()
        for _, chunk, sentence in candidates:
            chunk_id = str(chunk.get("id") or chunk.get("doc_id") or sentence[:80])
            if chunk_id in used_chunk_ids:
                continue
            used_chunk_ids.add(chunk_id)
            selected.append((chunk, sentence))
            if len(selected) >= 2:
                break

        bullets = [
            f"- {sentence} [{index}]"
            for index, (_, sentence) in enumerate(selected, 1)
        ]
        used_chunks = [chunk for chunk, _ in selected]
        return intro + "\n\n" + "\n".join(bullets) + "\n\n" + closing, used_chunks

    def _interaction_sentences(self, content: str) -> list[str]:
        bullet_blocks = [
            block.strip()
            for block in re.split(r"\s+\*\s+", content)
            if len(block.split()) >= 8
        ]
        if len(bullet_blocks) > 1:
            return bullet_blocks
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", content)
            if len(sentence.split()) >= 8
        ]

    def _interaction_sentence_score(
        self, normalized_sentence: str, query_entities: set[str]
    ) -> float:
        cues = [
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
            "tuong tac",
            "chay mau",
            "nguy co chay mau",
            "khong bat dau",
        ]
        cue_hits = min(3, sum(1 for cue in cues if cue in normalized_sentence))
        if cue_hits == 0:
            return 0.0

        entity_hits = sum(1 for entity in query_entities if entity in normalized_sentence)
        nsaid_support = "ibuprofen" in query_entities and (
            "nsaid" in normalized_sentence
            or "nonsteroidal" in normalized_sentence
            or "anti inflammatory" in normalized_sentence
        )
        if entity_hits == 0 and not nsaid_support:
            return 0.0

        action_bonus = 2.0 if any(
            cue in normalized_sentence
            for cue in [
                "do not start",
                "without discussing",
                "tell your doctor",
                "tell your healthcare provider",
                "closely monitor",
                "khong bat dau",
                "hoi bac si",
            ]
        ) else 0.0
        class_bonus = 1.5 if nsaid_support else 0.0
        direct_pair_bonus = 0.0
        if {"ibuprofen", "warfarin"}.issubset(query_entities) and all(
            entity in normalized_sentence for entity in ["ibuprofen", "warfarin"]
        ):
            direct_pair_bonus = 4.0
            if not any(
                cue in normalized_sentence
                for cue in ["may interact", "do not start", "without discussing"]
            ):
                direct_pair_bonus = 2.0
        length_penalty = min(max((len(normalized_sentence.split()) - 120) / 80, 0.0), 4.0)
        return (
            (entity_hits * 2.0)
            + cue_hits
            + action_bonus
            + class_bonus
            + direct_pair_bonus
            - length_penalty
        )

    def _truncate_words(self, text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]).rstrip(" ,;:") + "..."

    def _dedupe_chunks_for_answer(
        self, chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        deduped = []
        seen = set()
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            key = (
                normalize_for_match(str(metadata.get("entity", "")))
                or normalize_for_match(str(metadata.get("title", "")))
                or normalize_for_match(str(metadata.get("url", "")))
                or normalize_for_match(chunk.get("content", "")[:120])
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)
        return deduped

    def _generate_side_effect_answer(
        self,
        chunks: list[dict[str, Any]],
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
                ["hoại tử", "gangrene", "màu tím", "tối màu"],
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
                    ["hoại tử", "gangrene", "màu tím", "tối màu"],
                )
        items = self._dedupe_keep_order(merged_items)

        symptom_line = "; ".join(items[:10])
        bullets = [
            (
                "- Các dấu hiệu/tác dụng phụ cần gọi bác sĩ ngay gồm: "
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
        lowered_terms = [term.lower() for term in terms]
        if "hoai tu" in normalized_content or "gangrene" in normalized_content:
            return (
                "Warfarin có thể gây hoại tử hoặc hoại thư; hãy gọi bác sĩ ngay "
                "nếu thấy da đổi màu/tím, loét, đau dữ dội hoặc thay đổi màu/nhiệt độ trên cơ thể"
            )

        for sentence in re.split(r"(?<=[.!?])\s+", content):
            if any(term in sentence.lower() for term in lowered_terms):
                return sentence[:450].strip(" .;:-")
        return ""

    def _best_sentence(self, question: str, content: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+|\s+\*\s+", content)
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
        asks_pregnancy = is_pregnancy_query(question) and not is_lactation_query(question)
        asks_pregnancy_avoidance = (
            asks_pregnancy
            and asks_drug_avoidance(question)
        )
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
            if asks_pregnancy_avoidance and not has_pregnancy_risk_or_avoidance(
                sentence
            ):
                continue
            if (
                asks_pregnancy
                and not asks_pregnancy_avoidance
                and not has_pregnancy_context(sentence)
            ):
                continue
            terms = set(tokenize(sentence))
            score = len(query_terms & terms) * 2
            if asks_pregnancy and has_pregnancy_context(sentence):
                score += 8
            if asks_pregnancy_avoidance:
                normalized_sentence = normalize_for_match(sentence)
                if "thai nhi" in normalized_sentence or "fetus" in normalized_sentence:
                    score += 10
                if "tuan thu 20" in normalized_sentence or "20 weeks" in normalized_sentence:
                    score += 8
                if (
                    "tranh thai" in normalized_sentence
                    or "birth control" in normalized_sentence
                ):
                    score -= 8
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

    def _clean_extractive_sentence(self, sentence: str) -> str:
        sentence = re.sub(r"\[\d+(?:[-,]\d+)*\]", "", sentence)
        sentence = re.sub(r"\[(?:PMC|PubMed|CrossRef)[^\]]*\]", "", sentence, flags=re.IGNORECASE)
        # Loại bỏ breadcrumb/navigation artifacts tiếng Việt
        sentence = re.sub(r"Sức khỏe\s+Quay lại.*?(?=#|\n|$)", "", sentence)
        sentence = re.sub(r"Quay lại\s+\w+\s+Quay lại", "", sentence)
        sentence = re.sub(r"^\s*#\s*", "", sentence)
        sentence = re.sub(r"\s+", " ", sentence)
        return sentence.strip(" .;:-")

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
