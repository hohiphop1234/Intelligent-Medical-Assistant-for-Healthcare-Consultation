# Part 5: Pipeline & Chainlit UI (~3h)

> Phụ thuộc: Part 1-4 phải xong trước

---

## 5.1 Main Pipeline Orchestrator

#### [NEW] `src/rag_pipeline.py`

**Kết nối tất cả components theo flow:**

```
Emergency? → Classify → Retrieve → Grade → (Crawl?) → Generate → Validate
```

```python
class MedicalRAGPipeline:
    def __init__(self):
        self.safety_guard = SafetyGuard(CATEGORIES_PATH)
        self.query_router = QueryRouter()
        self.embedding_manager = DualEmbeddingManager()
        self.vector_store = VectorStore(CHROMA_PERSIST_DIR)
        self.bm25_store = BM25Store()
        self.hybrid_retriever = HybridRetriever(
            self.embedding_manager, self.vector_store, self.bm25_store
        )
        self.evidence_grader = EvidenceGrader()
        self.web_crawler = WebCrawler()
        self.response_generator = ResponseGenerator()
        self.response_validator = ResponseValidator()
        
        # Load BM25 index
        self.bm25_store.load("models/bm25_index.pkl")
    
    def process_query(self, question: str) -> dict:
        # 0. Detect language
        language = self.embedding_manager.detect_language(question)
        
        # 1. Emergency check (FIRST, before everything)
        if self.safety_guard.is_emergency(question):
            return self.safety_guard.emergency_response(question, language)
        
        # 2. Classify query
        classification = self.query_router.classify(question, language)
        
        # 3. Scope check
        if classification.category == "out_of_scope":
            return self.safety_guard.out_of_scope_response(question, language)
        
        if classification.confidence < CONFIDENCE_THRESHOLD:
            return self.safety_guard.insufficient_evidence_response(question, language)
        
        # 4. Hybrid retrieval (vector + BM25)
        results = self.hybrid_retriever.search(
            question, language=language, top_k=TOP_K,
            category_filter=classification.category if classification.confidence > 0.7 else None,
        )
        
        # 5. Grade evidence (Corrective RAG)
        graded = self.evidence_grader.grade(question, results)
        
        # 6. If insufficient → try web crawl
        if graded.needs_crawl:
            crawled = self.web_crawler.search(question, classification.entities)
            if crawled:
                # Embed crawled chunks on-the-fly
                for chunk in crawled:
                    chunk["embedding"] = self.embedding_manager.embed(chunk["content"], language)
                # Re-grade with combined evidence
                graded = self.evidence_grader.grade(question, results + crawled)
        
        # 7. Final evidence check
        if not graded.relevant_chunks:
            return self.safety_guard.insufficient_evidence_response(question, language)
        
        # 8. Generate response with LLM
        response = self.response_generator.generate(
            question, graded.relevant_chunks, classification
        )
        response["language"] = language
        response["confidence"] = graded.confidence
        
        # 9. Validate response
        validated = self.response_validator.validate(response, graded.relevant_chunks)
        
        return validated
    
    def ingest_data(self):
        """Ingest cleaned data into stores"""
        from src.data_cleaner import DataCleaner
        cleaner = DataCleaner()
        
        # Clean raw data
        cleaner.process_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR)
        
        # Ingest into vector + BM25
        for lang in ["en", "vi"]:
            chunks = load_jsonl(f"{PROCESSED_DATA_DIR}/chunks_{lang}.jsonl")
            texts = [c["content"] for c in chunks]
            embeddings = self.embedding_manager.embed_batch(texts, language=lang)
            self.vector_store.add_documents(chunks, embeddings, language=lang)
            self.bm25_store.build_index(chunks, language=lang)
        
        self.bm25_store.save("models/bm25_index.pkl")
        return self.vector_store.get_stats()
```

---

## 5.2 Chainlit UI

#### [NEW] `app.py`

```python
import chainlit as cl
from src.rag_pipeline import MedicalRAGPipeline

pipeline = MedicalRAGPipeline()

# === Welcome starters ===
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="💊 Tác dụng phụ Warfarin",
            message="Tác dụng phụ của thuốc Warfarin là gì?",
            icon="/public/pill.svg",
        ),
        cl.Starter(
            label="⚠️ Tương tác thuốc",
            message="Warfarin có tương tác với Ibuprofen không?",
            icon="/public/warning.svg",
        ),
        cl.Starter(
            label="🏥 Triệu chứng tiểu đường",
            message="Triệu chứng của bệnh tiểu đường type 2 là gì?",
            icon="/public/hospital.svg",
        ),
        cl.Starter(
            label="🤰 Thuốc khi mang thai",
            message="Phụ nữ mang thai có uống được Acetaminophen không?",
            icon="/public/pregnancy.svg",
        ),
    ]

# === Chat handler ===
@cl.on_message
async def on_message(message: cl.Message):
    result = pipeline.process_query(message.content)
    
    # --- Emergency / Out-of-scope / Insufficient evidence ---
    if result.get("type") in ["emergency", "out_of_scope", "insufficient_evidence"]:
        await cl.Message(content=result["message"]).send()
        return
    
    # --- Normal RAG response with streaming ---
    # Create source elements
    source_elements = []
    for src in result.get("sources", []):
        source_elements.append(cl.Text(
            name=f"[{src['index']}] {src['title']}",
            content=(
                f"**Source:** {src['source']}\n"
                f"**URL:** {src.get('url', 'N/A')}\n"
                f"**Section:** {src.get('section', 'N/A')}\n"
                f"**Relevance:** {src['score']:.2f}"
            ),
            display="side",
        ))
    
    # Risk badge
    risk_badges = {
        "low": "🟢 Low Risk",
        "medium": "🟡 Medium Risk",
        "high": "🔴 High Risk",
        "critical": "🚨 Critical",
    }
    risk_badge = risk_badges.get(result.get("risk_level", "medium"), "")
    confidence = result.get("confidence", "")
    
    # Stream response
    msg = cl.Message(content="", elements=source_elements)
    
    # Add risk badge header
    header = f"**{risk_badge}** | Confidence: {confidence}\n\n"
    await msg.stream_token(header)
    
    # Stream LLM response
    answer_stream = result.get("answer_stream")
    full_answer = ""
    if answer_stream:
        for chunk in answer_stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_answer += token
                await msg.stream_token(token)
    
    # Add disclaimer at end
    disclaimer = result.get("disclaimer", "")
    if disclaimer:
        await msg.stream_token(f"\n\n---\n{disclaimer}")
    
    await msg.send()

# === Chat start ===
@cl.on_chat_start
async def on_chat_start():
    stats = pipeline.vector_store.get_stats()
    await cl.Message(
        content=f"🏥 Medical Assistant sẵn sàng!\n"
                f"📊 Dataset: {stats['vi_count']} chunks VI + {stats['en_count']} chunks EN\n"
                f"🤖 LLM: Qwen 3.6 via OpenRouter\n"
                f"🔍 Search: Hybrid (Vector + BM25)"
    ).send()
```

#### [NEW] `chainlit.md`

```markdown
# 🏥 Intelligent Medical Assistant

Chào mừng bạn đến với **Trợ lý Y tế Thông minh**.

Tôi có thể giúp bạn về:
- 💊 Thông tin thuốc và tác dụng phụ
- ⚠️ Tương tác thuốc
- 🏥 Thông tin bệnh lý
- 🤰 Thuốc trong thai kỳ
- 👶 Nhi khoa
- 👴 Chăm sóc người cao tuổi

⚕️ **Lưu ý:** Thông tin chỉ mang tính tham khảo, không thay thế tư vấn y tế chuyên nghiệp.
```

---

## 5.3 CLI Entry Point

#### [NEW] `main.py`

```python
import argparse
from src.rag_pipeline import MedicalRAGPipeline

def main():
    parser = argparse.ArgumentParser(description="Medical RAG Assistant")
    parser.add_argument("--ingest", action="store_true", help="Ingest data into stores")
    parser.add_argument("--query", type=str, help="Query the assistant")
    args = parser.parse_args()
    
    pipeline = MedicalRAGPipeline()
    
    if args.ingest:
        print("📥 Ingesting data...")
        stats = pipeline.ingest_data()
        print(f"✅ Done: {stats}")
    
    elif args.query:
        result = pipeline.process_query(args.query)
        print(result.get("answer", result.get("message", "")))
    
    else:
        # Interactive mode
        print("🏥 Medical Assistant (type 'quit' to exit)")
        while True:
            q = input("\n❓ You: ").strip()
            if q.lower() in ("quit", "exit"):
                break
            result = pipeline.process_query(q)
            print(f"\n🤖 Assistant: {result.get('answer', result.get('message', ''))}")

if __name__ == "__main__":
    main()
```

---

## Verification cho Part 5

```bash
# 1. Test CLI
python main.py --ingest
python main.py --query "Tác dụng phụ của Warfarin?"

# 2. Test interactive mode
python main.py

# 3. Launch Chainlit UI
chainlit run app.py

# 4. Test in browser (http://localhost:8000):
#    - Click starter question
#    - Ask in-scope question → expect answer with citations + sources panel
#    - Ask out-of-scope → expect polite refusal
#    - Ask emergency → expect emergency response
#    - Ask in Vietnamese → expect Vietnamese response
#    - Ask in English → expect English response
```

**Xong Part 5 → chuyển sang [Part 6](file:///C:/Users/votru/.gemini/antigravity/brain/72dfbf26-8c80-4ebc-85dd-2bf3db7b1dca/plan_part6_evaluation.md)**
