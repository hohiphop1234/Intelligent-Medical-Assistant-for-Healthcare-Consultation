from __future__ import annotations

import re
from typing import Any

from src.utils import normalize_for_match

EMERGENCY_TRIGGERS = [
    r"dau nguc", r"kho tho", r"dot quy", r"nhoi mau", r"tu tu", 
    r"chay mau", r"bat tinh", r"co giat", r"soc phan ve", r"hon me", 
    r"qua lieu", r"ngo doc", r"ngat xiu"
]

EMERGENCY_RESPONSE = (
    "**⚠️ CẢNH BÁO CẤP CỨU ⚠️**\n\n"
    "Triệu chứng bạn mô tả có thể là tình huống cấp cứu y tế.\n\n"
    "**👉 Hãy gọi cấp cứu ngay:**\n\n"
    "- 🚑 **Cấp cứu toàn quốc: 115**\n"
    "- 🏥 BV Bạch Mai (Hà Nội): 024 3869 3731\n"
    "- 🏥 BV Chợ Rẫy (TP.HCM): 028 3855 4137\n"
    "- 🏥 BV Tâm Anh (HN): 024 3872 3872\n"
    "- 🏥 BV Tâm Anh (HCM): 028 7102 6789\n"
    "- 🏥 BV Vinmec (HN): 024 3974 3556\n\n"
    "⚕️ Vui lòng tìm trợ giúp y tế trực tiếp ngay."
)

OUT_OF_SCOPE_RESPONSE = (
    "Xin lỗi, tôi là trợ lý y tế. Câu hỏi này nằm ngoài phạm vi kiến thức của tôi. "
    "Bạn có thể hỏi tôi về triệu chứng bệnh, thông tin thuốc hoặc điều trị."
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

    def __init__(self, llm_client=None):
        self.triggers = [re.compile(rf"\b{p}\b", re.IGNORECASE) for p in EMERGENCY_TRIGGERS]
        self.llm = llm_client

    def is_emergency(self, query: str) -> bool:
        q_norm = normalize_for_match(query)
        
        # 1. Lọc thô bằng Regex
        if not any(t.search(q_norm) for t in self.triggers):
            return False
            
        # 2. Nếu không có LLM client -> mặc định return True (fall back to old behavior if needed)
        if not self.llm:
            return True
            
        # 3. LLM verify (Zero-shot)
        prompt = """
Bạn là chuyên gia y tế. Đọc câu hỏi của người dùng và xác định xem đây có phải là TÌNH HUỐNG CẤP CỨU Y TẾ KHẨN CẤP hay không.
- Nếu người dùng đang kể triệu chứng nguy hiểm xảy ra với họ/người thân -> Trả lời YES.
- Nếu họ chỉ hỏi kiến thức chung chung (dấu hiệu nhận biết, nguyên nhân, cách phòng) -> Trả lời NO.

Chỉ trả lời đúng 1 chữ: YES hoặc NO.
"""
        try:
            response = self.llm.generate_answer(query, system_prompt=prompt, max_new_tokens=512)
            if "Lỗi" in response or "Exception" in response:
                return True  # Fallback to emergency warning if LLM fails
            return "YES" in response.upper()
        except Exception:
            return True

    def emergency_response(self, query: str) -> dict[str, Any]:
        return {
            "type": "emergency",
            "message": EMERGENCY_RESPONSE,
            "risk_level": "critical",
            "requires_human": True,
            "language": "vi",
        }

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
