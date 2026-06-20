import os
import torch
from config import LORA_CHECKPOINT_DIR

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    PeftModel = None
    BitsAndBytesConfig = None


class QwenMedicalLLM:
    """
    Class quản lý và tải mô hình Qwen3-4B kết hợp với trọng số LoRA y tế.
    Mục đích: Cung cấp nhánh General QA trả lời các câu hỏi cơ bản siêu tốc cục bộ
    mà không cần thông qua OpenAI hay OpenRouter API.
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
            self.tokenizer = None
            self.base_model_name = "unsloth/Qwen3-4B-unsloth-bnb-4bit"

    def load_model(self):
        """Tải mô hình gốc 4-bit và ghép với LoRA Adapter"""
        if self._is_loaded or AutoModelForCausalLM is None:
            return

        print(f"[*] Đang tải mô hình gốc {self.base_model_name} (4-bit)...")
        # Cấu hình lượng tử hóa (Quantization) 4-bit để tiết kiệm VRAM
        try:
            # 1. Tải Base Model
            # Mô hình unsloth-bnb-4bit đã có sẵn quantization config, việc ép device_map={"": 0} sẽ tránh việc transformers tự đẩy sang CPU gây lỗi
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                device_map={"": 0},
                trust_remote_code=True
            )

            # 2. Gắn LoRA Adapter
            lora_path = os.path.abspath(LORA_CHECKPOINT_DIR)
            print(f"[*] Đang gắn LoRA Adapter từ {lora_path}...")
            self.model = PeftModel.from_pretrained(base_model, lora_path)
            
            # Đặt model ở chế độ suy luận (Inference)
            self.model.eval()
            self._is_loaded = True
            print("[+] Hoàn tất tải Qwen Medical LLM!")
        except Exception as e:
            print(f"[-] Lỗi tải mô hình LLM cục bộ: {e}")
            self.model = None
            self.tokenizer = None
            self._is_loaded = True  # Đánh dấu đã tải (thất bại) để không thử lại liên tục

    def generate_answer(self, question: str, max_new_tokens: int = 512, system_prompt: str = None) -> str:
        """
        Sinh câu trả lời dựa trên câu hỏi đầu vào.
        """
        if not self._is_loaded:
            self.load_model()

        if self.model is None or self.tokenizer is None:
            return "Lỗi: Không thể tải mô hình LLM cục bộ (thiếu thư viện hoặc cấu hình)."

        if system_prompt is None:
            system_prompt = "You are a medical question answering assistant. Answer clearly, cautiously, and remind users to consult healthcare professionals for personal medical decisions."

        # Tạo prompt chuẩn xác theo định dạng chat template mà model đã train
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Chuyển input lên GPU (nếu có)
        device = self.model.device
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        # Sinh câu trả lời
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Bỏ đi phần prompt trong output để chỉ lấy câu trả lời
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Lọc bỏ nội dung thừa nếu model sinh ra tag <think>
        if "<think>" in response and "</think>" in response:
            response = response.split("</think>")[-1].strip()
            
        return response.strip()
