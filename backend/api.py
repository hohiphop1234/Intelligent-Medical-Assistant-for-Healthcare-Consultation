from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical Assistant API")

# Setup CORS cho ReactJS frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        from src.langgraph_pipeline import LangGraphPipeline
        pipeline = LangGraphPipeline()
    return pipeline

@app.on_event("startup")
async def startup_event():
    logger.info("Preloading pipeline on startup...")
    get_pipeline()
    logger.info("Pipeline preloaded successfully.")

class ChatRequest(BaseModel):
    message: str
    isEmergency: Optional[bool] = False

class ChatResponse(BaseModel):
    type: str # "message", "emergency", "out_of_scope", "insufficient_evidence"
    message: str
    sources: Optional[List[Dict[str, Any]]] = None
    category: Optional[str] = None
    risk_level: Optional[str] = None
    route: Optional[str] = None
    confidence: Optional[float] = None

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        pipeline = get_pipeline()
        result = pipeline.process_query(request.message, isEmergency=request.isEmergency)
        
        # Xử lý các luồng đặc biệt
        if result.get("type") in {"out_of_scope", "insufficient_evidence"}:
            return ChatResponse(
                type=result["type"],
                message=result["message"]
            )
        if result.get("type") == "emergency":
            return ChatResponse(
                type="emergency",
                message=result.get("message", result.get("answer", "")),
                sources=result.get("sources", []),
                category=result.get("category", "overdose_triage"),
                risk_level="critical",
                route=result.get("route", "emergency_rag"),
                confidence=result.get("confidence", 1.0)
            )
        
        # Ghép phần disclaimer vào cuối answer
        answer = result.get("answer", "")
        if result.get("disclaimer"):
            answer += "\n\n---\n" + result["disclaimer"]
            
        return ChatResponse(
            type="message",
            message=answer,
            sources=result.get("sources", []),
            category=result.get("category", "unknown"),
            risk_level=result.get("risk_level", "medium"),
            route=result.get("route", "rag"),
            confidence=result.get("confidence", 0.0)
        )
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def stats_endpoint():
    try:
        pipeline = get_pipeline()
        stats = pipeline.rag_pipeline.vector_store.get_stats()
        return {"stats": stats}
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import StreamingResponse
import json

@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    pipeline = get_pipeline()
    
    async def event_generator():
        try:
            for item in pipeline.stream_query(request.message, isEmergency=request.isEmergency):
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Error in stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
