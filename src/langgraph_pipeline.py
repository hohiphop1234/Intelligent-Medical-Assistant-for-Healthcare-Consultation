from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, END

from config import CONFIDENCE_THRESHOLD
from src.rag_pipeline import MedicalRAGPipeline, FILTERABLE_CATEGORIES
from src.qwen_llm import QwenMedicalLLM
from src.topic_relevance import asks_drug_avoidance, should_route_pregnancy_to_drug_safety


# =====================================================================
# 1. Định nghĩa Trạng thái (State) của Đồ thị
# =====================================================================
class MedicalState(TypedDict):
    """
    State chung cho toàn bộ quá trình trả lời câu hỏi y tế.
    """
    question: str
    
    # Kết quả từ Classifier
    category: Optional[str]
    risk_level: Optional[str]
    entities: Optional[list[str]]
    confidence: Optional[float]
    
    # Quyết định điều hướng
    route: Optional[str]
    
    # Kết quả đầu ra
    answer: Optional[str]
    sources: Optional[list[dict[str, Any]]]
    safety_alert: Optional[bool]
    
    # Extra RAG Info
    evidence_score: Optional[float]
    retrieved_count: Optional[int]
    relevant_count: Optional[int]


# =====================================================================
# 2. Xây dựng Đồ thị LangGraph
# =====================================================================
class LangGraphPipeline:
    """
    Đồ thị điều phối luồng xử lý câu hỏi y khoa:
    Question -> Safety Node -> Classifier Node -> LLM QA Node / RAG Node -> END
    """
    
    def __init__(self):
        # Tái sử dụng các thành phần từ RAG Pipeline hiện tại
        self.rag_pipeline = MedicalRAGPipeline()
        self.safety_guard = self.rag_pipeline.safety_guard
        self.query_router = self.rag_pipeline.query_router
        self.embedding_manager = self.rag_pipeline.embedding_manager
        
        # Nhánh LLM General QA
        self.llm = QwenMedicalLLM()
        
        # Khởi tạo đồ thị
        self.graph = self._build_graph()
        
    def _build_graph(self):
        workflow = StateGraph(MedicalState)
        
        # Đăng ký các nodes
        workflow.add_node("safety_check_node", self.safety_check_node)
        workflow.add_node("classifier_node", self.classifier_node)
        workflow.add_node("general_qa_node", self.general_qa_node)
        workflow.add_node("rag_node", self.rag_node)
        
        # Đăng ký các edges
        workflow.set_entry_point("safety_check_node")
        
        # Từ safety_check_node, nếu bị chặn (safety_alert = True) -> END. Nếu an toàn -> classifier_node
        workflow.add_conditional_edges(
            "safety_check_node",
            self._route_after_safety,
            {
                "end": END,
                "continue": "classifier_node"
            }
        )
        
        # Từ classifier_node, phân nhánh: rag_node hoặc general_qa_node
        workflow.add_conditional_edges(
            "classifier_node",
            self._route_after_classifier,
            {
                "rag": "rag_node",
                "general_qa": "general_qa_node"
            }
        )
        
        # Cả hai nhánh đều dẫn tới kết thúc
        workflow.add_edge("general_qa_node", END)
        workflow.add_edge("rag_node", END)
        
        return workflow.compile()

    # =====================================================================
    # 3. Định nghĩa các Node (Thực thi logic)
    # =====================================================================
    
    def safety_check_node(self, state: MedicalState) -> MedicalState:
        """Kiểm tra câu hỏi có phải cấp cứu hay không"""
        question = state["question"]
        
        if self.safety_guard.is_emergency(question):
            emergency_resp = self.safety_guard.emergency_response(question)
            return {
                **state,
                "answer": emergency_resp["message"],
                "safety_alert": True
            }
            
        return {
            **state,
            "safety_alert": False
        }
        
    def classifier_node(self, state: MedicalState) -> MedicalState:
        """Phân loại câu hỏi để quyết định đi nhánh nào"""
        question = state["question"]
        
        classification = self.query_router.classify(question)
        
        # Xử lý out of scope hoặc thiếu dữ kiện
        if classification.category == "out_of_scope":
            out_resp = self.safety_guard.out_of_scope_response(question)
            return {
                **state,
                "category": classification.category,
                "risk_level": classification.risk_level,
                "route": "end_now",
                "answer": out_resp["message"]
            }
            
        if classification.confidence < CONFIDENCE_THRESHOLD:
            insuf_resp = self.safety_guard.insufficient_evidence_response(question)
            return {
                **state,
                "category": classification.category,
                "risk_level": classification.risk_level,
                "route": "end_now",
                "answer": insuf_resp["message"]
            }
            
        # Quyết định route: Các danh mục liên quan tới bệnh án, rủi ro cao -> RAG. Cơ bản -> LLM QA.
        # Ở đây chúng ta sẽ ép các risk categories vào RAG
        rag_categories = {
            "safety", "interactions", "contraindications", "contraindication",
            "pregnancy", "overdose", "pediatric", "patient_query", "case_based", "edge_case"
        }
        
        if classification.risk_level in ["high", "critical"] or classification.category in rag_categories:
            route = "rag"
        else:
            route = "general_qa"
            
        return {
            **state,
            "category": classification.category,
            "risk_level": classification.risk_level,
            "entities": classification.entities,
            "confidence": classification.confidence,
            "route": route
        }

    def general_qa_node(self, state: MedicalState) -> MedicalState:
        """Nhánh LLM: Dùng mô hình Qwen3-4B cục bộ để sinh câu trả lời"""
        # Nếu đã có answer từ safety check hoặc classifier (out of scope) thì bỏ qua LLM
        if state.get("answer"):
            return state
            
        question = state["question"]
        
        # Sinh câu trả lời siêu tốc bằng model cục bộ
        answer = self.llm.generate_answer(question)
        
        return {
            **state,
            "answer": answer,
            "sources": []  # Nhánh QA cơ bản không có trích dẫn RAG
        }
        
    def rag_node(self, state: MedicalState) -> MedicalState:
        """Nhánh RAG: Gọi logic RAG gốc của MedicalRAGPipeline"""
        # Tránh việc phân loại lại, chúng ta tận dụng cấu trúc cũ
        # Để nhanh, chúng ta chỉ gọi các step tìm kiếm và sinh của RAG pipeline cũ
        # Nhưng để tái sử dụng toàn diện nhất, có thể gọi .process_query (nó sẽ làm lại việc safety check một chút)
        
        # Gọi trực tiếp process_query của MedicalRAGPipeline (đã được bọc lại an toàn)
        result = self.rag_pipeline.process_query(state["question"])
        
        return {
            **state,
            "answer": result.get("answer"),
            "sources": result.get("sources", []),
            "evidence_score": result.get("evidence_score"),
            "retrieved_count": result.get("retrieved_count"),
            "relevant_count": result.get("relevant_count")
        }

    # =====================================================================
    # 4. Routing Functions
    # =====================================================================
    
    def _route_after_safety(self, state: MedicalState) -> str:
        if state.get("safety_alert"):
            return "end"
        return "continue"
        
    def _route_after_classifier(self, state: MedicalState) -> str:
        if state.get("route") == "end_now":
            # Gộp thành END nếu đã trả lời xong (Out of scope/insufficient)
            return "general_qa" # Sẽ bypass do đã có answer, hoặc có thể return rag_node. Tốt nhất tạo luồng chuẩn.
            # Ở đây LangGraph yêu cầu trả về key chính xác trong map đã định nghĩa.
            # Trong _build_graph map = {"rag": "rag_node", "general_qa": "general_qa_node"}
            # Nếu end_now, có thể ép vào general_qa_node, node đó phải check nếu có answer rồi thì bỏ qua sinh LLM.
        return state.get("route", "general_qa")

    # =====================================================================
    # 5. Hàm thực thi chính
    # =====================================================================
    
    def process_query(self, question: str) -> dict:
        """Entrypoint cho toàn bộ đồ thị"""
        initial_state = {
            "question": question,
            "category": None,
            "risk_level": None,
            "entities": None,
            "confidence": None,
            "route": None,
            "answer": None,
            "sources": None,
            "safety_alert": False,
            "evidence_score": None,
            "retrieved_count": None,
            "relevant_count": None
        }
        
        final_state = self.graph.invoke(initial_state)
        
        # Nếu là safety alert hoặc out_of_scope/insufficient_evidence
        if final_state.get("safety_alert"):
            return {"type": "emergency", "message": final_state.get("answer")}
        
        if final_state.get("route") == "end_now":
            if final_state.get("category") == "out_of_scope":
                return {"type": "out_of_scope", "message": final_state.get("answer")}
            else:
                return {"type": "insufficient_evidence", "message": final_state.get("answer")}
        
        # Format lại output giống với RAG Pipeline cũ để app.py không bị lỗi
        return {
            "answer": final_state.get("answer", "Xin lỗi, đã có lỗi xảy ra."),
            "sources": final_state.get("sources", []),
            "risk_level": final_state.get("risk_level", "low"),
            "category": final_state.get("category", "unknown"),
            "route": final_state.get("route", "rag"),
            "classification_confidence": final_state.get("confidence", 0.0),
            "confidence": final_state.get("evidence_score", 0.0),
            "evidence_score": final_state.get("evidence_score", 0.0),
            "retrieved_count": final_state.get("retrieved_count", 0),
            "relevant_count": final_state.get("relevant_count", 0)
        }
