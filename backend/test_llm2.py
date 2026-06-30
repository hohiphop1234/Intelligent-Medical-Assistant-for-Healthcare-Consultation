from src.qwen_llm import QwenMedicalLLM
llm = QwenMedicalLLM()
prompt = """
Bạn là chuyên gia y tế. Đọc câu hỏi của người dùng và xác định xem đây có phải là TÌNH HUỐNG CẤP CỨU Y TẾ KHẨN CẤP hay không.
- Nếu người dùng đang kể triệu chứng nguy hiểm xảy ra với họ/người thân -> Trả lời YES.
- Nếu họ chỉ hỏi kiến thức chung chung (dấu hiệu nhận biết, nguyên nhân, cách phòng) -> Trả lời NO.

Chỉ trả lời đúng 1 chữ: YES hoặc NO.
"""
query = "tôi đang bị tức ngực, khó thở quá"
print("Generating answer...")
resp = llm.generate_answer(query, system_prompt=prompt, max_new_tokens=10)
print(f"RESPONSE: {resp}")
print("YES in response?", "YES" in resp.upper())
