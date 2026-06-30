from __future__ import annotations

import logging
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
        self.llm = QwenMedicalLLM()

    def rewrite(self, question: str, entities: list[str] | None = None) -> str:
        """
        Rewrite the query using local Qwen. 
        If it fails or returns an invalid/empty result, fallback to the original query.
        """
        # Fast path: bypass LLM if query is already short/concise
        words = question.strip().split()
        if len(words) <= 4:
            logger.info(f"[Query Rewriter] Query is already short ({len(words)} words). Bypassing rewrite.")
            return question
        prompt = f"User question: {question}\n"
        if entities:
            prompt += f"Detected medical entities: {', '.join(entities)}\n"
        prompt += "Optimized search query:"

        try:
            # Tăng max_new_tokens lên 256 phòng trường hợp model vẫn sinh <think>
            rewritten = self.llm.generate_answer(
                question=prompt,
                max_new_tokens=256,
                system_prompt=REWRITE_SYSTEM_PROMPT
            )
            rewritten = rewritten.strip().strip('"').strip("'")
            
            # Kiểm tra nếu câu trả lời hợp lệ (không trống, không phải câu lỗi, không phải câu xin lỗi)
            if (rewritten and 
                not rewritten.startswith("Lỗi:") and 
                not rewritten.startswith("Xin lỗi") and
                len(rewritten) < 150): # Query thực tế không được quá dài
                
                logger.info(f"[Query Rewriter] Rewrote '{question}' -> '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.error(f"[Query Rewriter] Error during rewrite: {e}")
        
        logger.info(f"[Query Rewriter] Fallback to original query: '{question}'")
        return question
