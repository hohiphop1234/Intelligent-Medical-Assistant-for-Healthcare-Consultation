from __future__ import annotations

import re
from typing import Any

from src.utils import load_json, normalize_for_match


EMERGENCY_PATTERNS = [
    r"\bdau nguc\b",
    r"\bkho tho\b",
    r"\bkhong tho duoc\b",
    r"\bdot quy\b",
    r"\bnhoi mau\b",
    r"\btu tu\b",
    r"\bmuon chet\b",
    r"\btu gay thuong tich\b",
    r"\bchay mau nhieu\b",
    r"\bchay mau khong ngung\b",
    r"\bbat tinh\b",
    r"\bco giat\b",
    r"\bdi ung\b.*\bsung\b",
    r"\bsoc phan ve\b",
    r"\bhon me\b",
]

EMERGENCY_RESPONSE = (
    "**CANH BAO KHAN CAP**\n\n"
    "Trieu chung ban mo ta co the la tinh huong cap cuu y te.\n\n"
    "Hay goi cap cuu 115 hoac den co so y te gan nhat ngay lap tuc. "
    "Neu dang o My, hay goi 911.\n\n"
    "Toi khong the thay the bac si trong tinh huong khan cap. "
    "Vui long tim tro giup y te truc tiep ngay."
)

OUT_OF_SCOPE_RESPONSE = (
    "Cau hoi nay nam ngoai pham vi kien thuc y te cua tro ly.\n\n"
    "Toi co the ho tro ve thuoc, tac dung phu, tuong tac thuoc, "
    "thong tin benh ly, thai ky, nhi khoa va nguoi cao tuoi."
)

INSUFFICIENT_EVIDENCE_RESPONSE = (
    "Tôi không tìm được đủ thông tin đáng tin cậy để trả lời câu hỏi này "
    "một cách an toàn.\n\n"
    "Vui lòng tham khảo bác sĩ, dược sĩ, hoặc nguồn y tế chính thống."
)

DISCLAIMERS = {
    "low": "Thông tin chỉ mang tính tham khảo. Hãy hỏi bác sĩ nếu cần.",
    "medium": (
        "Thông tin chỉ mang tính tham khảo và không thay thế tư vấn y tế "
        "chuyên nghiệp. Vui lòng tham khảo ý kiến bác sĩ."
    ),
    "high": (
        "QUAN TRỌNG: Không tự ý thay đổi liều thuốc hoặc phác đồ điều trị. "
        "Hãy hỏi bác sĩ hoặc dược sĩ trước khi hành động."
    ),
    "critical": (
        "CẢNH BÁO: Đây là nhóm thông tin nhạy cảm. Bắt buộc tham khảo "
        "bác sĩ chuyên khoa trước khi áp dụng."
    ),
}


def get_disclaimer(risk_level: str) -> str:
    return DISCLAIMERS.get(risk_level, DISCLAIMERS["medium"])


class SafetyGuard:
    """Safety checks that run before and after retrieval."""

    def __init__(self, categories_path: str):
        self.categories = load_json(categories_path)["categories"]
        self.emergency_patterns = [re.compile(pattern) for pattern in EMERGENCY_PATTERNS]

    def is_emergency(self, query: str) -> bool:
        query_normalized = normalize_for_match(query)
        for pattern in self.emergency_patterns:
            if pattern.search(query_normalized):
                return True
        return False

    def emergency_response(self, query: str) -> dict[str, Any]:
        return {
            "type": "emergency",
            "message": EMERGENCY_RESPONSE,
            "risk_level": "critical",
            "requires_human": True,
            "language": "vi",
        }

    def is_medical_scope(self, query: str) -> tuple[bool, float]:
        query_normalized = normalize_for_match(query)
        max_score = 0.0
        for category in self.categories:
            keywords = category.get("keywords_vi", [])
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
            keywords = category.get("keywords_vi", [])
            matches = sum(
                1 for keyword in keywords if normalize_for_match(keyword) in query_normalized
            )
            score = matches / max(len(keywords), 1)
            if score > best[0]:
                best = (score, category)
        return best[1]

    def out_of_scope_response(self, query: str) -> dict[str, Any]:
        return {
            "type": "out_of_scope",
            "message": OUT_OF_SCOPE_RESPONSE,
            "risk_level": "none",
            "language": "vi",
        }

    def insufficient_evidence_response(self, query: str) -> dict[str, Any]:
        return {
            "type": "insufficient_evidence",
            "message": INSUFFICIENT_EVIDENCE_RESPONSE,
            "risk_level": "medium",
            "language": "vi",
        }
