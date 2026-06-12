from __future__ import annotations

import re
from typing import Any

from src.utils import load_json, normalize_for_match


EMERGENCY_PATTERNS = {
    "en": [
        r"\bchest pain\b",
        r"\bsevere chest pain\b",
        r"\bcan t breathe\b",
        r"\bdifficulty breathing\b",
        r"\bstroke\b",
        r"\bheart attack\b",
        r"\boverdose\b",
        r"\btook too many\b",
        r"\bsuicid",
        r"\bwant to die\b",
        r"\bself harm\b",
        r"\bsevere bleeding\b",
        r"\bunconscious\b",
        r"\bseizure\b",
        r"\ballergic reaction\b.*\bswelling\b",
        r"\banaphyla",
        r"\bpoisoning\b",
        r"\bnot responsive\b",
    ],
    "vi": [
        r"\bdau nguc\b",
        r"\bkho tho\b",
        r"\bkhong tho duoc\b",
        r"\bdot quy\b",
        r"\bnhoi mau\b",
        r"\bqua lieu\b",
        r"\buong qua nhieu\b",
        r"\btu tu\b",
        r"\bmuon chet\b",
        r"\btu gay thuong tich\b",
        r"\bchay mau nhieu\b",
        r"\bchay mau khong ngung\b",
        r"\bbat tinh\b",
        r"\bco giat\b",
        r"\bdi ung\b.*\bsung\b",
        r"\bsoc phan ve\b",
        r"\bngo doc\b",
        r"\bhon me\b",
    ],
}

EMERGENCY_RESPONSES = {
    "vi": (
        "**CANH BAO KHAN CAP**\n\n"
        "Trieu chung ban mo ta co the la tinh huong cap cuu y te.\n\n"
        "Hay goi cap cuu 115 hoac den co so y te gan nhat ngay lap tuc. "
        "Neu dang o My, hay goi 911.\n\n"
        "Toi khong the thay the bac si trong tinh huong khan cap. "
        "Vui long tim tro giup y te truc tiep ngay."
    ),
    "en": (
        "**EMERGENCY WARNING**\n\n"
        "The symptoms you describe may indicate a medical emergency.\n\n"
        "Call emergency services (911) or go to the nearest hospital immediately. "
        "In Vietnam, call 115.\n\n"
        "I cannot replace a doctor in emergency situations. "
        "Please seek immediate medical attention."
    ),
}

OUT_OF_SCOPE_RESPONSES = {
    "vi": (
        "Cau hoi nay nam ngoai pham vi kien thuc y te cua tro ly.\n\n"
        "Toi co the ho tro ve thuoc, tac dung phu, tuong tac thuoc, "
        "thong tin benh ly, thai ky, nhi khoa va nguoi cao tuoi."
    ),
    "en": (
        "This question is outside the assistant's medical knowledge scope.\n\n"
        "I can help with drug information, side effects, drug interactions, "
        "disease information, pregnancy, pediatric care, and elderly care."
    ),
}

INSUFFICIENT_EVIDENCE_RESPONSES = {
    "vi": (
        "Toi khong tim duoc du thong tin dang tin cay de tra loi cau hoi nay "
        "mot cach an toan.\n\n"
        "Vui long tham khao bac si, duoc si, hoac nguon y te chinh thong."
    ),
    "en": (
        "I could not find enough reliable evidence to answer this question safely.\n\n"
        "Please consult a doctor, pharmacist, or a trusted medical source."
    ),
}

DISCLAIMERS = {
    "low": {
        "vi": "Thông tin chỉ mang tính tham khảo. Hãy hỏi bác sĩ nếu cần.",
        "en": "For informational purposes only. Consult a doctor if needed.",
    },
    "medium": {
        "vi": (
            "Thông tin chỉ mang tính tham khảo và không thay thế tư vấn y tế "
            "chuyên nghiệp. Vui lòng tham khảo ý kiến bác sĩ."
        ),
        "en": (
            "This information is for reference only and does not replace "
            "professional medical advice. Please consult a doctor."
        ),
    },
    "high": {
        "vi": (
            "QUAN TRỌNG: Không tự ý thay đổi liều thuốc hoặc phác đồ điều trị. "
            "Hãy hỏi bác sĩ hoặc dược sĩ trước khi hành động."
        ),
        "en": (
            "IMPORTANT: Do not change dosage or treatment on your own. "
            "Consult your doctor or pharmacist before making changes."
        ),
    },
    "critical": {
        "vi": (
            "CẢNH BÁO: Đây là nhóm thông tin nhạy cảm. Bắt buộc tham khảo "
            "bác sĩ chuyên khoa trước khi áp dụng."
        ),
        "en": (
            "WARNING: This involves sensitive medical information. You should "
            "consult a qualified clinician before acting on it."
        ),
    },
}


def get_disclaimer(risk_level: str, language: str) -> str:
    language = "vi" if language == "vi" else "en"
    return DISCLAIMERS.get(risk_level, DISCLAIMERS["medium"])[language]


class SafetyGuard:
    """Safety checks that run before and after retrieval."""

    def __init__(self, categories_path: str):
        self.categories = load_json(categories_path)["categories"]
        self.emergency_patterns = {
            language: [re.compile(pattern) for pattern in patterns]
            for language, patterns in EMERGENCY_PATTERNS.items()
        }

    def is_emergency(self, query: str) -> bool:
        query_normalized = normalize_for_match(query)
        for patterns in self.emergency_patterns.values():
            for pattern in patterns:
                if pattern.search(query_normalized):
                    return True
        return False

    def emergency_response(self, query: str, language: str) -> dict[str, Any]:
        language = "vi" if language == "vi" else "en"
        return {
            "type": "emergency",
            "message": EMERGENCY_RESPONSES[language],
            "risk_level": "critical",
            "requires_human": True,
            "language": language,
        }

    def is_medical_scope(self, query: str) -> tuple[bool, float]:
        query_normalized = normalize_for_match(query)
        max_score = 0.0
        for category in self.categories:
            keywords = category.get("keywords_en", []) + category.get("keywords_vi", [])
            if not keywords:
                continue
            matches = sum(
                1 for keyword in keywords if normalize_for_match(keyword) in query_normalized
            )
            score = matches / max(len(keywords), 1)
            max_score = max(max_score, score)
        return max_score >= 0.05, max_score

    def best_category(self, query: str) -> dict[str, Any] | None:
        query_normalized = normalize_for_match(query)
        best: tuple[float, dict[str, Any] | None] = (0.0, None)
        for category in self.categories:
            keywords = category.get("keywords_en", []) + category.get("keywords_vi", [])
            matches = sum(
                1 for keyword in keywords if normalize_for_match(keyword) in query_normalized
            )
            score = matches / max(len(keywords), 1)
            if score > best[0]:
                best = (score, category)
        return best[1]

    def out_of_scope_response(self, query: str, language: str) -> dict[str, Any]:
        language = "vi" if language == "vi" else "en"
        return {
            "type": "out_of_scope",
            "message": OUT_OF_SCOPE_RESPONSES[language],
            "risk_level": "none",
            "language": language,
        }

    def insufficient_evidence_response(self, query: str, language: str) -> dict[str, Any]:
        language = "vi" if language == "vi" else "en"
        return {
            "type": "insufficient_evidence",
            "message": INSUFFICIENT_EVIDENCE_RESPONSES[language],
            "risk_level": "medium",
            "language": language,
        }
