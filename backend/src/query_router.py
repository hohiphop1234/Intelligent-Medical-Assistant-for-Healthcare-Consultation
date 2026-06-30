from __future__ import annotations

import json
from dataclasses import dataclass

@dataclass
class QueryClassification:
    intent: str
    category: str
    entities: list[str]
    risk_level: str
    confidence: float
    requires_rag: bool

class QueryRouter:
    """Classify medical queries before retrieval using LLM."""

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def classify(self, query: str) -> QueryClassification:
        if not self.llm:
            return QueryClassification(
                intent="drug_query",
                category="drug_query",
                entities=[],
                risk_level="high",
                confidence=0.9,
                requires_rag=True
            )

        prompt = """
Bạn là hệ thống định tuyến (router) cho một trợ lý y tế AI.
Hãy phân loại câu hỏi sau thành ĐÚNG MỘT TRONG BA nhóm:
1. "faq": Các câu hỏi chào hỏi (hello, xin chào), khả năng của bạn (bạn làm được gì, bạn là ai), hoặc lời cảm ơn.
2. "drug_query": Tất cả các câu hỏi liên quan đến sức khỏe, triệu chứng bệnh, điều trị, thông tin thuốc, y tế.
3. "out_of_scope": Các câu hỏi ngoài luồng, KHÔNG liên quan đến y tế hoặc trợ lý y tế (ví dụ: viết code, nấu ăn, thời tiết, điện thoại nào tốt).

Trả về kết quả dưới định dạng JSON hợp lệ với 1 key duy nhất là "category".
Ví dụ: {"category": "faq"} hoặc {"category": "drug_query"} hoặc {"category": "out_of_scope"}
"""
        try:
            # Tùy thuộc vào LLM client, parse JSON
            response = self.llm.generate_answer(query, system_prompt=prompt, max_new_tokens=512)
            # Find the JSON part in the response
            start_idx = response.find("{")
            end_idx = response.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx+1]
                data = json.loads(json_str)
                category = data.get("category", "drug_query").lower()
            else:
                category = "drug_query"
                
            if category not in ["faq", "out_of_scope"]:
                category = "drug_query"
        except Exception:
            category = "drug_query"

        return QueryClassification(
            intent=category,
            category=category,
            entities=[],
            risk_level="high" if category == "drug_query" else "low",
            confidence=0.9,
            requires_rag=(category == "drug_query")
        )
