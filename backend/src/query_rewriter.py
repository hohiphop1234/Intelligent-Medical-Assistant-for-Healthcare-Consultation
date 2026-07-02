from __future__ import annotations

import logging
from pathlib import Path
from src.qwen_llm import QwenMedicalLLM

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """You are a medical search query optimization assistant.
Your task is to rewrite the user's question into a concise search query (in Vietnamese) that is optimized for retrieval from a medical database (Vector Search + BM25).

Core Rules:
1. Translate English drug names or medical terms to common Vietnamese equivalents if they are frequently searched in Vietnamese, or keep both to maximize retrieval (e.g., "Paracetamol / Acetaminophen").
2. Extract the core entities (drug names, diseases, symptoms) and make them prominent.
3. Keep the query concise (under 10-15 words). Do not include conversational filler words like "cho tôi hỏi", "là gì", "có sao không", "nhờ bác sĩ tư vấn".
4. Output ONLY the optimized search query. Do not write any thinking process, explanation, intro, or markdown formatting. Output the query directly.
5. If the query is already concise and medical-oriented, output it as-is.

Examples:
- "uống panadol với efferalgan có bị quá liều không bác sĩ" -> "quá liều panadol efferalgan paracetamol"
- "tôi đang mang thai tuần 24 uống thuốc cảm cúm tiffy được không" -> "tiffy phụ nữ mang thai thai kỳ tuần 24"
- "thuốc metformin tương tác với insulin như thế nào" -> "tương tác metformin insulin"
"""

class QueryRewriter:
    """Rewrite raw medical queries into optimized keywords for retrieval."""

    def __init__(self):
        self.t5_model_path = Path(__file__).resolve().parent.parent / "models" / "t5_query_rewriter"
        self.t5_model = None
        self.t5_tokenizer = None
        
        # Nạp mô hình T5 đã fine-tune nếu tồn tại trong thư mục models/t5_query_rewriter
        if self.t5_model_path.exists():
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                logger.info(f"[Query Rewriter] Loading fine-tuned T5 model from {self.t5_model_path}...")
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.t5_tokenizer = AutoTokenizer.from_pretrained(str(self.t5_model_path))
                self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(str(self.t5_model_path)).to(self.device)
                self.t5_model.eval()
                logger.info("[Query Rewriter] T5 query rewriter loaded successfully!")
            except Exception as e:
                logger.warning(f"[Query Rewriter] Failed to load T5 model ({e}), will fallback to Qwen LLM.")

        self.llm = QwenMedicalLLM()

    def rewrite(self, question: str, entities: list[str] | None = None) -> str:
        """
        Rewrite the query using fine-tuned T5 model (ultra-fast ~0.05s).
        If T5 is not available or fails, fallback to local Qwen LLM.
        """
        # Fast path: bypass if query is already very short
        words = question.strip().split()
        if len(words) <= 4:
            logger.info(f"[Query Rewriter] Query is already short ({len(words)} words). Bypassing rewrite.")
            return question

        # 🚀 Ưu tiên 1: Sử dụng mô hình T5 đã Fine-tune chuyên biệt
        if self.t5_model and self.t5_tokenizer:
            try:
                import torch
                prefix = "tối ưu từ khóa tìm kiếm y khoa: "
                input_text = prefix + question
                inputs = self.t5_tokenizer(input_text, max_length=128, truncation=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.t5_model.generate(
                        inputs.input_ids,
                        max_new_tokens=48,
                        num_beams=2,
                        repetition_penalty=1.5,
                        no_repeat_ngram_size=2
                    )
                rewritten = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                if rewritten and len(rewritten) > 2:
                    logger.info(f"[Query Rewriter - T5] Rewrote '{question}' -> '{rewritten}'")
                    return rewritten
            except Exception as e:
                logger.error(f"[Query Rewriter - T5] Error: {e}")

        # 🚀 Ưu tiên 2: Fallback về prompt LLM Qwen
        prompt = f"User question: {question}\n"
        if entities:
            prompt += f"Detected medical entities: {', '.join(entities)}\n"
        prompt += "Optimized search query:"

        try:
            rewritten = self.llm.generate_answer(
                question=prompt,
                max_new_tokens=256,
                system_prompt=REWRITE_SYSTEM_PROMPT
            )
            rewritten = rewritten.strip().strip('"').strip("'")
            
            if (rewritten and 
                not rewritten.startswith("Lỗi:") and 
                not rewritten.startswith("Xin lỗi") and
                len(rewritten) < 150):
                
                logger.info(f"[Query Rewriter - Qwen] Rewrote '{question}' -> '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.error(f"[Query Rewriter] Error during rewrite: {e}")
        
        logger.info(f"[Query Rewriter] Fallback to original query: '{question}'")
        return question
