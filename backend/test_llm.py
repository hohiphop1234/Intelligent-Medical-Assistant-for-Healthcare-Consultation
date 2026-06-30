import sys
from src.qwen_llm import QwenMedicalLLM

llm = QwenMedicalLLM()
query = "tôi đang bị tức ngực, khó thở quá"
prompt = f"""
Bạn là chuyên gia y tế. Đọc câu hỏi của người dùng và xác định xem đây có phải là TÌNH HUỐNG CẤP CỨU Y TẾ KHẨN CẤP hay không.
- Nếu người dùng đang kể triệu chứng nguy hiểm xảy ra với họ/người thân -> Trả lời YES.
- Nếu họ chỉ hỏi kiến thức chung chung (dấu hiệu nhận biết, nguyên nhân, cách phòng) -> Trả lời NO.

Câu hỏi: "{query}"
Chỉ trả lời đúng 1 chữ: YES hoặc NO.
"""
print("QWEN_OUTPUT:")
print(repr(llm.invoke(prompt)))
