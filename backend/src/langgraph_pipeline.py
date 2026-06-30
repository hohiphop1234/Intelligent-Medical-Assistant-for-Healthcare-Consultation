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
    isEmergency: Optional[bool]  # Biến cờ cấp cứu từ người dùng bấm bật lên
    
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
# 2. Đồ thị riêng biệt cho nhánh Emergency (Emergency RAG Graph)
# =====================================================================
class EmergencyRAGGraph:
    """
    Đồ thị riêng biệt xử lý nhánh cấp cứu (Emergency RAG Branch).
    Tạo đồ thị mới để tách biệt hoàn toàn nhánh emergency, tránh làm rối đồ thị chính.
    """
    def __init__(self, rag_pipeline: MedicalRAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(MedicalState)
        workflow.add_node("emergency_rag_node", self.emergency_rag_node)
        workflow.set_entry_point("emergency_rag_node")
        workflow.add_edge("emergency_rag_node", END)
        return workflow.compile()

    def emergency_rag_node(self, state: MedicalState) -> MedicalState:
        """Chuyển thẳng qua node RAG của emergency: Truy xuất kiến thức cấp cứu và sơ cứu"""
        question = state["question"]
        result = self.rag_pipeline.process_emergency_query(question)
        return {
            **state,
            "answer": result.get("answer"),
            "sources": result.get("sources", []),
            "category": result.get("category", "overdose_triage"),
            "risk_level": result.get("risk_level", "critical"),
            "route": "emergency_rag",
            "evidence_score": result.get("evidence_score", 1.0),
            "retrieved_count": result.get("retrieved_count", 0),
            "relevant_count": result.get("relevant_count", 0),
        }


# =====================================================================
# 3. Xây dựng Đồ thị LangGraph chính
# =====================================================================
class LangGraphPipeline:
    """
    Đồ thị điều phối luồng xử lý câu hỏi y khoa:
    Router (isEmergency flag) -> Emergency Graph / Classifier Node -> RAG Node / General QA -> END
    """
    
    def __init__(self):
        self.rag_pipeline = MedicalRAGPipeline()
        self.safety_guard = self.rag_pipeline.safety_guard
        self.query_router = self.rag_pipeline.query_router
        self.embedding_manager = self.rag_pipeline.embedding_manager
        
        # Nhánh LLM General QA
        self.llm = QwenMedicalLLM()
        
        # Khởi tạo đồ thị riêng cho Emergency
        self.emergency_graph = EmergencyRAGGraph(self.rag_pipeline)
        
        # Khởi tạo đồ thị chính
        self.graph = self._build_graph()
        
    def _build_graph(self):
        workflow = StateGraph(MedicalState)
        
        # Đăng ký các nodes
        workflow.add_node("router_node", self.router_node)
        workflow.add_node("emergency_branch_node", self.emergency_branch_node)
        workflow.add_node("classifier_node", self.classifier_node)
        workflow.add_node("general_qa_node", self.general_qa_node)
        workflow.add_node("rag_node", self.rag_node)
        
        # Đăng ký entry point tại router_node kiểm tra state
        workflow.set_entry_point("router_node")
        
        # Từ router_node, kiểm tra biến isEmergency trong state (không hardcode từ khóa)
        workflow.add_conditional_edges(
            "router_node",
            self._route_initial,
            {
                "emergency": "emergency_branch_node",
                "standard": "classifier_node"
            }
        )
        
        # Nhánh emergency chuyển thẳng sang đồ thị Emergency RAG riêng biệt rồi kết thúc
        workflow.add_edge("emergency_branch_node", END)
        
        # Từ classifier_node, phân nhánh: rag_node hoặc general_qa_node
        workflow.add_conditional_edges(
            "classifier_node",
            self._route_after_classifier,
            {
                "rag": "rag_node",
                "general_qa": "general_qa_node"
            }
        )
        
        workflow.add_edge("general_qa_node", END)
        workflow.add_edge("rag_node", END)
        
        return workflow.compile()

    # =====================================================================
    # 4. Định nghĩa các Node (Thực thi logic)
    # =====================================================================
    
    def router_node(self, state: MedicalState) -> MedicalState:
        """Kiểm tra cờ isEmergency từ state, không dùng hardcoded keywords"""
        return state

    def emergency_branch_node(self, state: MedicalState) -> MedicalState:
        """Gọi thực thi đồ thị riêng biệt của nhánh emergency"""
        return self.emergency_graph.graph.invoke(state)
        
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
        """Nhánh LLM: Dùng mô hình Qwen cục bộ để sinh câu trả lời"""
        if state.get("answer"):
            return state
            
        question = state["question"]
        answer = self.llm.generate_answer(question)
        
        return {
            **state,
            "answer": answer,
            "sources": []
        }
        
    def rag_node(self, state: MedicalState) -> MedicalState:
        """Nhánh RAG tiêu chuẩn"""
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
    # 5. Routing Functions
    # =====================================================================
    
    def _route_initial(self, state: MedicalState) -> str:
        """Điều hướng dựa vào biến isEmergency trong state (bật cờ từ UI/API)"""
        if state.get("isEmergency") is True or state.get("is_emergency") is True:
            return "emergency"
        return "standard"
        
    def _route_after_classifier(self, state: MedicalState) -> str:
        if state.get("route") == "end_now":
            return "general_qa"
        return state.get("route", "general_qa")

    # =====================================================================
    # 6. Hàm thực thi chính & streaming
    # =====================================================================
    
    def process_query(self, question: str, isEmergency: bool = False) -> dict:
        """Entrypoint cho toàn bộ đồ thị"""
        initial_state = {
            "question": question,
            "isEmergency": isEmergency,
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
        
        if final_state.get("route") == "emergency_rag" or final_state.get("safety_alert"):
            return {
                "type": "emergency",
                "message": final_state.get("answer", ""),
                "answer": final_state.get("answer", ""),
                "sources": final_state.get("sources", []),
                "risk_level": final_state.get("risk_level", "critical"),
                "category": final_state.get("category", "overdose_triage"),
                "route": "emergency_rag",
                "classification_confidence": 1.0,
                "confidence": final_state.get("evidence_score", 1.0),
                "evidence_score": final_state.get("evidence_score", 1.0),
                "retrieved_count": final_state.get("retrieved_count", 0),
                "relevant_count": final_state.get("relevant_count", 0)
            }
        
        if final_state.get("route") == "end_now":
            if final_state.get("category") == "out_of_scope":
                return {"type": "out_of_scope", "message": final_state.get("answer")}
            else:
                return {"type": "insufficient_evidence", "message": final_state.get("answer")}
        
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

    def stream_query(self, question: str, isEmergency: bool = False):
        """Stream qua RAG pipeline theo nhánh tương ứng"""
        if isEmergency:
            for item in self.rag_pipeline.stream_emergency_query(question):
                yield item
        else:
            for item in self.rag_pipeline.stream_query(question):
                yield item
