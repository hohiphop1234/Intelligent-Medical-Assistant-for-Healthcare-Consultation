import requests
import json
import os
from config import LLAMA_SERVER_URL

class QwenMedicalLLM:
    """
    Class quản lý và gọi mô hình Qwen3-4B thông qua llama-server.
    Mục đích: Cung cấp nhánh General QA và RAG chạy siêu tốc cục bộ thông qua API tương thích OpenAI của llama-server.
    """
    _instance = None

    def __new__(cls):
        # Đảm bảo Singleton pattern
        if cls._instance is None:
            cls._instance = super(QwenMedicalLLM, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.llama_url = f"{LLAMA_SERVER_URL}/v1/chat/completions"
        self.model_name = "qwen3-4b-thinking" # Có thể bất kỳ tên nào vì llama-server dùng model đã nạp sẵn
        
        # Kiểm tra trạng thái llama-server
        try:
            res = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=2)
            self._is_loaded = res.status_code == 200
        except Exception:
            self._is_loaded = False

    def load_model(self):
        # Không cần load thủ công, llama-server tự quản lý
        pass

    def generate_answer(self, question: str, max_new_tokens: int = 1024, system_prompt: str = None) -> str:
        """
        Sinh câu trả lời dựa trên câu hỏi đầu vào thông qua llama-server API.
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
            "max_tokens": max_new_tokens,
            "temperature": 0.15,
            "top_p": 0.9,
            "frequency_penalty": 1.15
        }
        
        try:
            response = requests.post(self.llama_url, json=payload, timeout=600)
            if response.status_code != 200:
                return f"Lỗi gọi llama-server API: {response.text}"
                
            data = response.json()
            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
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
            return "Lỗi: Không thể kết nối tới llama-server. Vui lòng đảm bảo llama-server đang chạy."
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Exception: {e}"
            
    def stream_answer(self, question: str, system_prompt: str = "", max_new_tokens: int = 2048):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "max_tokens": max_new_tokens,
            "temperature": 0.15,
            "top_p": 0.9,
            "frequency_penalty": 1.15
        }
        
        import json
        try:
            response = requests.post(self.llama_url, json=payload, stream=True, timeout=600)
            if response.status_code != 200:
                yield f"Lỗi gọi llama-server API: {response.text}"
                return
                
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        json_str = line[6:]
                        if json_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(json_str)
                            chunk = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"\n[Lỗi kết nối llama-server: {e}]"
