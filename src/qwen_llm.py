import os
from config import BASE_DIR

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

class QwenMedicalLLM:
    """
    Class quản lý và tải mô hình Qwen3-4B (GGUF).
    Mục đích: Cung cấp nhánh General QA và RAG chạy siêu tốc cục bộ.
    """
    _instance = None
    _is_loaded = False

    def __new__(cls):
        # Đảm bảo Singleton pattern để tránh load model nhiều lần vào RAM
        if cls._instance is None:
            cls._instance = super(QwenMedicalLLM, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._is_loaded:
            self.model = None

    def load_model(self):
        """Tải mô hình GGUF bằng llama.cpp"""
        if self._is_loaded or Llama is None:
            return

        gguf_path = os.path.join(BASE_DIR, "models", "qwen3-4b-thinking.gguf")
        print(f"[*] Đang tải mô hình GGUF từ {gguf_path}...")
        
        if not os.path.exists(gguf_path):
            print(f"[-] Không tìm thấy file GGUF tại {gguf_path}!")
            self._is_loaded = True
            return

        try:
            # Tải model GGUF
            self.model = Llama(
                model_path=gguf_path,
                n_gpu_layers=-1, # Offload toàn bộ lên GPU (RTX 3050)
                n_ctx=4096,      # Tăng độ dài ngữ cảnh lên 4096 để RAG thoải mái nhồi context
                verbose=False    # Tắt spam log
            )
            self._is_loaded = True
            print("[+] Hoàn tất tải Qwen Medical LLM (GGUF)!")
        except Exception as e:
            print(f"[-] Lỗi tải mô hình LLM cục bộ (GGUF): {e}")
            self.model = None
            self._is_loaded = True

    def generate_answer(self, question: str, max_new_tokens: int = 1024, system_prompt: str = None) -> str:
        """
        Sinh câu trả lời dựa trên câu hỏi đầu vào.
        """
        if not self._is_loaded:
            self.load_model()

        if self.model is None:
            if Llama is None:
                return "Lỗi: Không tìm thấy thư viện `llama-cpp-python`."
            return "Lỗi: Không thể tải mô hình GGUF."

        if system_prompt is None:
            system_prompt = "You are a medical question answering assistant. Answer clearly, cautiously, and remind users to consult healthcare professionals for personal medical decisions."

        # Tạo prompt chuẩn xác
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        try:
            # Sinh chữ dùng llama.cpp
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens, # Tăng lên để đủ chỗ cho phần <think>
                temperature=0.4,
                top_p=0.9,
                repeat_penalty=1.15 # Ngăn chặn lỗi vòng lặp từ (VD: "nổi ban đỏ, nổi ban đỏ...")
            )
            
            # Lấy chuỗi kết quả
            answer = response["choices"][0]["message"]["content"]
            
            # Lọc bỏ nội dung thừa nếu model sinh ra tag <think>
            if "</think>" in answer:
                parts = answer.split("</think>")
                if len(parts) > 1 and parts[-1].strip():
                    answer = parts[-1].strip()
                else:
                    pass
            elif answer.startswith("<think>"):
                # Nếu bị cắt ngang do hết max_tokens và không có </think>
                answer = "Xin lỗi, câu trả lời bị ngắt quãng do vượt quá giới hạn độ dài. Vui lòng thử hỏi ngắn gọn hơn."
                
            # Xóa sạch tag <think> nếu có rớt lại
            answer = answer.replace("<think>", "").strip()
            
            return answer
        except Exception as e:
            return f"Lỗi sinh chữ: {e}"

