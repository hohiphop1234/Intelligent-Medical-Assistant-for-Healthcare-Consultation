from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, END

from config import CONFIDENCE_THRESHOLD, TOP_K
from src.rag_pipeline import MedicalRAGPipeline
from src.qwen_llm import QwenMedicalLLM
from src.query_rewriter import QueryRewriter
from src.query_router import QueryClassification
from src.safety_guard import get_disclaimer


# =====================================================================
# 1. Định nghĩa Trạng thái (State) của Đồ thị
# =====================================================================
class MedicalState(TypedDict):
    """
    State chung cho toàn bộ quá trình trả lời câu hỏi y tế.
    """
    question: str
    rewritten_query: Optional[str]
    
    # Kết quả từ Classifier
    category: Optional[str]
    risk_level: Optional[str]
    entities: Optional[list[str]]
    confidence: Optional[float]
    
    # Quyết định điều hướng
    route: Optional[str]
    exit_type: Optional[str]  # "emergency" | "out_of_scope" | "insufficient_evidence" | None
    
    # Kết quả RAG
    retrieved_chunks: Optional[list[dict[str, Any]]]
    relevant_chunks: Optional[list[dict[str, Any]]]
    evidence_score: Optional[float]
    evidence_confidence: Optional[str]
    needs_crawl: Optional[bool]
    crawl_attempted: bool
    
    # Kết quả đầu ra
    answer: Optional[str]
    sources: Optional[list[dict[str, Any]]]
    disclaimer: Optional[str]
    
    # Hậu kiểm
    validation_issues: Optional[list[str]]
    is_valid: Optional[bool]


# =====================================================================
# 2. Xây dựng Đồ thị LangGraph
# =====================================================================
class LangGraphPipeline:
    """
    Đồ thị điều phối luồng xử lý câu hỏi y khoa theo kiến trúc mới:
    Risk Triage -> Intent Router -> Query Rewrite -> Hybrid Retrieval ->
    Evidence Grading -> (Low Score -> Trusted Search -> Evidence Grading) -> 
    Answer Gen / LLM QA -> Medical Validation -> END
    """
    
    def __init__(self):
        # Nhánh LLM General QA & Query Rewrite
        self.llm = QwenMedicalLLM()
        self.query_rewriter = QueryRewriter()
        
        # Tái sử dụng các thành phần từ RAG Pipeline hiện tại
        self.rag_pipeline = MedicalRAGPipeline(llm_client=self.llm)
        self.safety_guard = self.rag_pipeline.safety_guard
        self.embedding_manager = self.rag_pipeline.embedding_manager
        self.hybrid_retriever = self.rag_pipeline.hybrid_retriever
        self.evidence_grader = self.rag_pipeline.evidence_grader
        self.web_crawler = self.rag_pipeline.web_crawler
        self.response_generator = self.rag_pipeline.response_generator
        self.response_validator = self.rag_pipeline.response_validator
        
        # Router: Khởi tạo với QwenMedicalLLM để chạy phân loại thực tế
        from src.query_router import QueryRouter
        self.query_router = QueryRouter(self.llm)
        
        # Khởi tạo đồ thị
        self.graph = self._build_graph()
        
    def _build_graph(self):
        workflow = StateGraph(MedicalState)
        
        # Đăng ký các nodes
        workflow.add_node("risk_triage", self.risk_triage_node)
        workflow.add_node("intent_router", self.intent_router_node)
        workflow.add_node("query_rewrite", self.query_rewrite_node)
        workflow.add_node("hybrid_retrieval", self.hybrid_retrieval_node)
        workflow.add_node("evidence_grading", self.evidence_grading_node)
        workflow.add_node("trusted_search", self.trusted_search_node)
        workflow.add_node("general_qa", self.general_qa_node)
        workflow.add_node("answer_generation", self.answer_generation_node)
        workflow.add_node("medical_validation", self.medical_validation_node)
        workflow.add_node("early_exit", self.early_exit_node)
        
        # Thiết lập điểm bắt đầu
        workflow.set_entry_point("risk_triage")
        
        # Edges từ risk_triage
        workflow.add_conditional_edges(
            "risk_triage",
            self._route_after_risk_triage,
            {
                "early_exit": "early_exit",
                "continue": "intent_router"
            }
        )
        
        # Edges từ intent_router
        workflow.add_conditional_edges(
            "intent_router",
            self._route_after_intent_router,
            {
                "early_exit": "early_exit",
                "general_qa": "general_qa",
                "query_rewrite": "query_rewrite"
            }
        )
        
        # Các bước tuần tự của RAG
        workflow.add_edge("query_rewrite", "hybrid_retrieval")
        workflow.add_edge("hybrid_retrieval", "evidence_grading")
        
        # Edges từ evidence_grading (rẽ nhánh dựa trên điểm evidence)
        workflow.add_conditional_edges(
            "evidence_grading",
            self._route_after_evidence_grading,
            {
                "low_score": "trusted_search",
                "high_score": "answer_generation",
                "early_exit": "early_exit"
            }
        )
        
        # Tìm kiếm bổ sung xong quay lại chấm điểm evidence mới
        workflow.add_edge("trusted_search", "evidence_grading")
        
        # Cả luồng RAG lẫn LLM QA đều đi qua hậu kiểm validation
        workflow.add_edge("general_qa", "medical_validation")
        workflow.add_edge("answer_generation", "medical_validation")
        
        # Điểm kết thúc
        workflow.add_edge("medical_validation", END)
        workflow.add_edge("early_exit", END)
        
        return workflow.compile()

    # =====================================================================
    # 3. Định nghĩa các Node (Thực thi logic)
    # =====================================================================
    
    def risk_triage_node(self, state: MedicalState) -> dict:
        """Kiểm tra xem câu hỏi có phải tình huống khẩn cấp y tế hay không."""
        question = state["question"]
        
        if self.safety_guard.is_emergency(question):
            emergency_resp = self.safety_guard.emergency_response(question)
            return {
                "answer": emergency_resp["message"],
                "risk_level": "critical",
                "route": "early_exit",
                "exit_type": "emergency"
            }
            
        return {"route": "continue"}
        
    def intent_router_node(self, state: MedicalState) -> dict:
        """Phân loại ý định câu hỏi để rẽ nhánh."""
        question = state["question"]
        classification = self.query_router.classify(question)
        
        # 1. Out of scope
        if classification.category == "out_of_scope":
            out_resp = self.safety_guard.out_of_scope_response(question)
            return {
                "category": classification.category,
                "risk_level": classification.risk_level,
                "route": "early_exit",
                "exit_type": "out_of_scope",
                "answer": out_resp["message"]
            }
            
        # 2. Thiếu độ tin cậy phân loại
        if classification.confidence < CONFIDENCE_THRESHOLD:
            insuf_resp = self.safety_guard.insufficient_evidence_response(question)
            return {
                "category": classification.category,
                "risk_level": classification.risk_level,
                "route": "early_exit",
                "exit_type": "insufficient_evidence",
                "answer": insuf_resp["message"]
            }
            
        # 3. FAQ / Greeting / Chào hỏi -> Gọi trực tiếp LLM
        if classification.category == "faq":
            return {
                "category": classification.category,
                "risk_level": classification.risk_level,
                "confidence": classification.confidence,
                "route": "faq"
            }
            
        # 4. Drug Query -> Đi qua luồng RAG
        return {
            "category": classification.category,
            "risk_level": classification.risk_level,
            "confidence": classification.confidence,
            "entities": classification.entities,
            "route": "drug_query"
        }
        
    def query_rewrite_node(self, state: MedicalState) -> dict:
        """Tối ưu hóa và viết lại câu hỏi tìm kiếm."""
        question = state["question"]
        entities = state.get("entities")
        rewritten_query = self.query_rewriter.rewrite(question, entities)
        return {"rewritten_query": rewritten_query}
        
    def hybrid_retrieval_node(self, state: MedicalState) -> dict:
        """Thực hiện tìm kiếm hybrid (Vector + BM25)."""
        query = state.get("rewritten_query") or state["question"]
        
        # Khởi dựng QueryClassification để tính toán top_k của retrieval
        classification = QueryClassification(
            intent=state.get("category", "drug_query"),
            category=state.get("category", "drug_query"),
            entities=state.get("entities") or [],
            risk_level=state.get("risk_level", "high"),
            confidence=state.get("confidence", 0.9),
            requires_rag=True
        )
        
        search_top_k = self.rag_pipeline._retrieval_top_k(query, classification)
        category_filter = self.rag_pipeline._retrieval_category_filter(query, classification)
        
        results = self.hybrid_retriever.search(
            query,
            top_k=search_top_k,
            category_filter=category_filter
        )
        return {"retrieved_chunks": results}
        
    def evidence_grading_node(self, state: MedicalState) -> dict:
        """Đánh giá chất lượng của evidence tìm được."""
        question = state["question"]
        chunks = state.get("retrieved_chunks") or []
        
        graded = self.evidence_grader.grade(question, chunks)
        
        return {
            "relevant_chunks": graded.relevant_chunks,
            "evidence_score": graded.score,
            "evidence_confidence": graded.confidence,
            "needs_crawl": graded.needs_crawl
        }
        
    def trusted_search_node(self, state: MedicalState) -> dict:
        """Tìm kiếm bổ sung trên web nếu KB không đủ thông tin."""
        question = state["question"]
        entities = state.get("entities") or []
        retrieved_chunks = state.get("retrieved_chunks") or []
        
        crawled = self.web_crawler.search(question, entities)
        if crawled:
            for chunk in crawled:
                chunk["embedding"] = self.embedding_manager.embed(chunk["content"])
            retrieved_chunks = retrieved_chunks + crawled
            
        return {
            "retrieved_chunks": retrieved_chunks,
            "crawl_attempted": True
        }
        
    def general_qa_node(self, state: MedicalState) -> dict:
        """Trả lời câu hỏi FAQ không cần thông tin RAG."""
        question = state["question"]
        answer = self.llm.generate_answer(question)
        return {
            "answer": answer,
            "sources": []
        }
        
    def answer_generation_node(self, state: MedicalState) -> dict:
        """Sinh câu trả lời tổng hợp từ các relevant chunks."""
        question = state["question"]
        chunks = state.get("relevant_chunks") or []
        
        classification = QueryClassification(
            intent=state.get("category", "drug_query"),
            category=state.get("category", "drug_query"),
            entities=state.get("entities") or [],
            risk_level=state.get("risk_level", "high"),
            confidence=state.get("confidence", 0.9),
            requires_rag=True
        )
        
        response = self.response_generator.generate(
            question,
            chunks,
            classification
        )
        
        return {
            "answer": response.get("answer"),
            "sources": response.get("sources") or []
        }
        
    def medical_validation_node(self, state: MedicalState) -> dict:
        """Hậu kiểm an toàn y tế và bổ sung disclaimer."""
        answer = state.get("answer") or ""
        chunks = state.get("relevant_chunks") or []
        risk_level = state.get("risk_level") or "medium"
        
        response = {
            "answer": answer,
            "risk_level": risk_level,
            "language": "vi",
            "category": state.get("category"),
            "route": state.get("route"),
            "exit_type": state.get("exit_type")
        }
        
        validated = self.response_validator.validate(response, chunks)
        
        return {
            "answer": validated.get("answer"),
            "disclaimer": validated.get("disclaimer"),
            "validation_issues": validated.get("validation_issues") or [],
            "is_valid": validated.get("is_valid", True)
        }
        
    def early_exit_node(self, state: MedicalState) -> dict:
        """Node gom để kết thúc sớm khi gặp điều kiện dừng khẩn cấp/ngoài luồng."""
        exit_type = state.get("exit_type")
        
        if exit_type == "emergency":
            return {}  # Đã có answer từ risk_triage
        elif exit_type == "out_of_scope":
            return {}  # Đã có answer từ intent_router
        elif exit_type == "insufficient_evidence" or not state.get("answer"):
            resp = self.safety_guard.insufficient_evidence_response(state["question"])
            return {
                "answer": resp["message"],
                "exit_type": "insufficient_evidence"
            }
            
        return {}

# =====================================================================
# 4. Routing Functions
# =====================================================================
    
    def _route_after_risk_triage(self, state: MedicalState) -> str:
        if state.get("route") == "early_exit":
            return "early_exit"
        return "continue"
        
    def _route_after_intent_router(self, state: MedicalState) -> str:
        route = state.get("route")
        if route == "early_exit":
            return "early_exit"
        elif route == "faq":
            return "general_qa"
        else:
            return "query_rewrite"
            
    def _route_after_evidence_grading(self, state: MedicalState) -> str:
        relevant = state.get("relevant_chunks") or []
        needs_crawl = state.get("needs_crawl", False)
        crawl_attempted = state.get("crawl_attempted", False)
        
        # 1. Có tài liệu tốt -> đi sinh câu trả lời
        if not needs_crawl and relevant:
            return "high_score"
            
        # 2. Tài liệu yếu và chưa crawl -> đi tìm kiếm bổ sung
        if needs_crawl and not crawl_attempted:
            return "low_score"
            
        # 3. Đã crawl mà vẫn có tài liệu -> đi sinh câu trả lời
        if relevant:
            return "high_score"
            
        # 4. Hoàn toàn không có tài liệu nào tốt -> kết thúc sớm
        return "early_exit"

    # =====================================================================
    # 5. Hàm thực thi chính
    # =====================================================================
    
    def process_query(self, question: str) -> dict:
        """Entrypoint cho toàn bộ đồ thị"""
        initial_state = {
            "question": question,
            "rewritten_query": None,
            "category": None,
            "risk_level": None,
            "entities": None,
            "confidence": None,
            "route": None,
            "exit_type": None,
            "retrieved_chunks": None,
            "relevant_chunks": None,
            "evidence_score": None,
            "evidence_confidence": None,
            "needs_crawl": None,
            "crawl_attempted": False,
            "answer": None,
            "sources": None,
            "disclaimer": None,
            "validation_issues": None,
            "is_valid": None
        }
        
        final_state = self.graph.invoke(initial_state)
        
        # Trả về định dạng lỗi đặc biệt nếu thoát sớm
        exit_type = final_state.get("exit_type")
        if exit_type in {"emergency", "out_of_scope", "insufficient_evidence"}:
            return {
                "type": exit_type,
                "message": final_state.get("answer")
            }
        
        # Format lại output giống với RAG Pipeline cũ để app.py/api.py hoạt động hoàn hảo
        return {
            "answer": final_state.get("answer", "Xin lỗi, đã có lỗi xảy ra."),
            "sources": final_state.get("sources", []),
            "risk_level": final_state.get("risk_level", "low"),
            "category": final_state.get("category", "unknown"),
            "route": final_state.get("route", "rag"),
            "classification_confidence": final_state.get("confidence", 0.0),
            "confidence": final_state.get("evidence_score", 0.0),
            "evidence_score": final_state.get("evidence_score", 0.0),
            "disclaimer": final_state.get("disclaimer", ""),
            "validation_issues": final_state.get("validation_issues", []),
            "is_valid": final_state.get("is_valid", True),
            "retrieved_count": len(final_state.get("retrieved_chunks") or []),
            "relevant_count": len(final_state.get("relevant_chunks") or [])
        }

    def stream_query(self, question: str):
        """Streaming entrypoint cho đồ thị LangGraph."""
        # 1. Khởi tạo state
        state = {
            "question": question,
            "rewritten_query": None,
            "category": None,
            "risk_level": None,
            "entities": None,
            "confidence": None,
            "route": None,
            "exit_type": None,
            "retrieved_chunks": None,
            "relevant_chunks": None,
            "evidence_score": None,
            "evidence_confidence": None,
            "needs_crawl": None,
            "crawl_attempted": False,
            "answer": None,
            "sources": None,
            "disclaimer": None,
            "validation_issues": None,
            "is_valid": None
        }
        
        # 2. Chạy Risk Triage
        triage_update = self.risk_triage_node(state)
        state.update(triage_update)
        if self._route_after_risk_triage(state) == "early_exit":
            exit_update = self.early_exit_node(state)
            state.update(exit_update)
            yield {"type": "metadata", "data": {"type": "emergency", "message": state["answer"]}}
            return
            
        # 3. Chạy Intent Router
        router_update = self.intent_router_node(state)
        state.update(router_update)
        route = self._route_after_intent_router(state)
        if route == "early_exit":
            exit_update = self.early_exit_node(state)
            state.update(exit_update)
            yield {"type": "metadata", "data": {"type": state["exit_type"], "message": state["answer"]}}
            return
            
        # 4. Phân nhánh FAQ vs RAG (Drug Query)
        if route == "general_qa":
            # Gửi metadata trước
            yield {
                "type": "metadata",
                "data": {
                    "type": "message",
                    "sources": [],
                    "category": state["category"],
                    "risk_level": state["risk_level"],
                    "route": "general_qa",
                    "disclaimer": get_disclaimer(state["risk_level"])
                }
            }
            # Stream câu trả lời FAQ từ LLM
            answer_parts = []
            for token in self.llm.stream_answer(state["question"]):
                answer_parts.append(token)
                yield {"type": "token", "content": token}
            
            # Chạy Validation node để kiểm tra câu trả lời
            state["answer"] = "".join(answer_parts)
            val_update = self.medical_validation_node(state)
            state.update(val_update)
            
            # Gửi thông báo an toàn nếu phát hiện vi phạm y tế nguy hiểm
            if not state["is_valid"]:
                issues = state.get("validation_issues", [])
                has_prohibited = any("Prohibited" in issue for issue in issues)
                if has_prohibited:
                    warning_text = "Phát hiện chỉ định điều trị hoặc chẩn đoán tự ý. Vui lòng tham khảo ý kiến bác sĩ chuyên khoa."
                else:
                    warning_text = "; ".join(issues)
                yield {"type": "token", "content": f"\n\n⚠️ **[Cảnh báo an toàn]**: {warning_text}"}
            return
            
        # Luồng RAG (Drug Query)
        # 5. Query Rewrite
        rewrite_update = self.query_rewrite_node(state)
        state.update(rewrite_update)
        
        # 6. Hybrid Retrieval
        retrieve_update = self.hybrid_retrieval_node(state)
        state.update(retrieve_update)
        
        # 7. Evidence Grading
        while True:
            grade_update = self.evidence_grading_node(state)
            state.update(grade_update)
            
            next_step = self._route_after_evidence_grading(state)
            if next_step == "low_score":
                # Tìm kiếm bổ sung (web crawl)
                crawl_update = self.trusted_search_node(state)
                state.update(crawl_update)
                # Tiếp tục vòng lặp để chấm điểm lại tài liệu mới crawl
                continue
            elif next_step == "early_exit":
                exit_update = self.early_exit_node(state)
                state.update(exit_update)
                yield {"type": "metadata", "data": {"type": "insufficient_evidence", "message": state["answer"]}}
                return
            else: # high_score
                break
                
        # 8. Sinh nguồn và gửi metadata trước cho client
        sources = []
        for i, chunk in enumerate(state["relevant_chunks"] or [], 1):
            metadata = chunk.get("metadata", {})
            sources.append({
                "index": i,
                "id": chunk.get("id", ""),
                "title": metadata.get("title") or chunk.get("title", "Unknown"),
                "content": chunk.get("content", "")
            })
            
        yield {
            "type": "metadata",
            "data": {
                "type": "message",
                "sources": sources,
                "category": state["category"],
                "risk_level": state["risk_level"],
                "route": "rag",
                "disclaimer": get_disclaimer(state["risk_level"])
            }
        }
        
        # 9. Stream câu trả lời RAG
        classification = QueryClassification(
            intent=state.get("category", "drug_query"),
            category=state.get("category", "drug_query"),
            entities=state.get("entities") or [],
            risk_level=state.get("risk_level", "high"),
            confidence=state.get("confidence", 0.9),
            requires_rag=True
        )
        
        answer_parts = []
        for token in self.response_generator.generate_stream(
            state["question"],
            state["relevant_chunks"],
            classification
        ):
            answer_parts.append(token)
            yield {"type": "token", "content": token}
            
        # 10. Chạy Validation node
        state["answer"] = "".join(answer_parts)
        val_update = self.medical_validation_node(state)
        state.update(val_update)
        
        if not state["is_valid"]:
            issues = state.get("validation_issues", [])
            has_prohibited = any("Prohibited" in issue for issue in issues)
            if has_prohibited:
                warning_text = "Phát hiện chỉ định điều trị hoặc chẩn đoán tự ý. Vui lòng tham khảo ý kiến bác sĩ chuyên khoa."
            else:
                warning_text = "; ".join(issues)
            yield {"type": "token", "content": f"\n\n⚠️ **[Cảnh báo an toàn]**: {warning_text}"}
