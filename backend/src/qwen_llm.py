import requests
import json
import os

class QwenMedicalLLM:
    """
    Class quản lý và gọi mô hình Qwen3-4B thông qua Ollama local.
    Mục đích: Cung cấp nhánh General QA và RAG chạy siêu tốc cục bộ thông qua API của Ollama.
    """
    _instance = None

    def __new__(cls):
        # Đảm bảo Singleton pattern
        if cls._instance is None:
            cls._instance = super(QwenMedicalLLM, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/chat"
        self.model_name = "qwen_medical" # Tên mô hình đã được import vào Ollama
        
        # Kiểm tra trạng thái Ollama
        try:
            res = requests.get("http://localhost:11434/")
            self._is_loaded = res.status_code == 200
        except Exception:
            self._is_loaded = False

    def load_model(self):
        # Không cần load thủ công, Ollama tự động quản lý vào RAM/VRAM
        pass

    def generate_answer(self, question: str, max_new_tokens: int = 1024, system_prompt: str = None) -> str:
        """
        Sinh câu trả lời dựa trên câu hỏi đầu vào thông qua Ollama API.
        """
        if system_prompt is None:
            system_prompt = "You are a medical question answering assistant. Answer clearly, cautiously, and remind users to consult healthcare professionals for personal medical decisions."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "num_ctx": 4096,
                "temperature": 0.15,  # Nhiệt độ rất thấp để ổn định output y tế
                "top_p": 0.9,
                "repeat_penalty": 1.15
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=600)
            if response.status_code != 200:
                if response.status_code == 404:
                    return "Lỗi: Không tìm thấy mô hình 'qwen_medical' trong Ollama. Vui lòng cài đặt và import mô hình bằng Modelfile."
                return f"Lỗi gọi Ollama API: {response.text}"
                
            data = response.json()
            answer = data.get("message", {}).get("content", "")
            
            # Xử lý dọn dẹp các luồng suy nghĩ <think> nếu mô hình sinh ra
            if "</think>" in answer:
                parts = answer.split("</think>")
                if parts[-1].strip():
                    answer = parts[-1].strip()
                else:
                    # Nếu không có gì sau </think>, nghĩa là nó chỉ sinh ra suy nghĩ rồi dừng
                    answer = parts[0].replace("<think>", "").strip()
                    if not answer:
                        answer = "Xin lỗi, câu trả lời bị trống. Vui lòng thử lại."
            elif answer.startswith("<think>"):
                answer = answer.replace("<think>", "").strip()
                if not answer:
                    answer = "Xin lỗi, hệ thống bị gián đoạn trong lúc suy nghĩ. Vui lòng thử lại."
            else:
                answer = answer.strip()
                
            if not answer:
                answer = "Xin lỗi, mô hình AI không tạo được câu trả lời cho câu hỏi này."
            return answer
            
        except requests.exceptions.ConnectionError:
            return "Lỗi: Không thể kết nối tới Ollama. Vui lòng đảm bảo Ollama đang chạy trên máy tính của bạn."
        except Exception as e:
            print(f"Lỗi gọi Ollama: {e}")
            return f"Lỗi: {str(e)}"
            
    def stream_answer(self, question: str, system_prompt: str = "", max_new_tokens: int = 2048):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "num_predict": max_new_tokens,
                "num_ctx": 6144,
                "temperature": 0.15,
                "top_p": 0.9,
                "repeat_penalty": 1.15
            }
        }
        
        import json
        try:
            response = requests.post(self.ollama_url, json=payload, stream=True, timeout=600)
            if response.status_code != 200:
                yield f"Lỗi gọi Ollama API: {response.text}"
                return
                
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
        except Exception as e:
            yield f"\n[Lỗi kết nối Ollama: {e}]"
