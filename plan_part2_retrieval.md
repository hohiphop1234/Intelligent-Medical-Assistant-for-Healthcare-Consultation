# Part 2: Embedding & Hybrid Retrieval (~3-4h)

> Phụ thuộc: Part 1 phải xong trước

---

## 2.1 Dual Embedding Manager

#### [NEW] `src/embeddings.py`

**2 model, auto-detect ngôn ngữ:**

| Model | Ngôn ngữ | Params | Dùng cho |
|-------|----------|--------|----------|
| `Dqdung205/medical_vietnamese_embedding` | VI | 0.3B | Chunks + queries tiếng Việt |
| `BAAI/bge-m3` | EN (multilingual) | 568M | Chunks + queries tiếng Anh |

```python
from sentence_transformers import SentenceTransformer
from langdetect import detect

class DualEmbeddingManager:
    def __init__(self):
        self.vi_model = SentenceTransformer("Dqdung205/medical_vietnamese_embedding")
        self.en_model = SentenceTransformer("BAAI/bge-m3")
        self._cache = {}  # {text_hash: embedding}
    
    def detect_language(self, text: str) -> str:
        try:
            return "vi" if detect(text) == "vi" else "en"
        except:
            return "en"
    
    def embed(self, text: str, language: str = None) -> list[float]:
        """Embed single text, auto-detect language if not specified"""
        if language is None:
            language = self.detect_language(text)
        
        cache_key = hash(f"{language}:{text}")
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if language == "vi":
            emb = self.vi_model.encode(text).tolist()
        else:
            emb = self.en_model.encode(text).tolist()
        
        self._cache[cache_key] = emb
        return emb
    
    def embed_batch(self, texts: list[str], language: str) -> list[list[float]]:
        """Batch embed, all same language"""
        model = self.vi_model if language == "vi" else self.en_model
        return model.encode(texts, show_progress_bar=True).tolist()
    
    def get_embedding_dim(self, language: str) -> int:
        """Return embedding dimension for collection creation"""
        if language == "vi":
            return self.vi_model.get_sentence_embedding_dimension()
        return self.en_model.get_sentence_embedding_dimension()
```

**Lưu ý quan trọng:**
- Lần đầu chạy sẽ download models từ HuggingFace (~2GB tổng)
- Cache embeddings vào memory để tránh re-compute
- Có thể thêm disk cache sau nếu cần

---

## 2.2 Vector Store

#### [NEW] `src/vector_store.py`

**ChromaDB với 2 collections** (vì 2 model embedding có dimension khác nhau):

```python
import chromadb

class VectorStore:
    def __init__(self, persist_dir: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_vi = self.client.get_or_create_collection(
            name="medical_rag_vi",
            metadata={"hnsw:space": "cosine"}
        )
        self.collection_en = self.client.get_or_create_collection(
            name="medical_rag_en",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, chunks: list[dict], embeddings: list, language: str):
        """Add chunks to appropriate collection"""
        collection = self.collection_vi if language == "vi" else self.collection_en
        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["content"] for c in chunks],
            embeddings=embeddings,
            metadatas=[{
                "source": c.get("source", ""),
                "entity": c.get("entity", ""),
                "category": c.get("topic_group", ""),
                "risk_level": c.get("risk_level", "medium"),
                "section": c.get("section", ""),
                "url": c.get("url", ""),
                "title": c.get("title", ""),
            } for c in chunks]
        )
    
    def search(self, query_embedding: list, language: str, top_k: int = 5, 
               category_filter: str = None) -> list[dict]:
        """Search with optional metadata filter"""
        collection = self.collection_vi if language == "vi" else self.collection_en
        
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "distances", "metadatas"]
        )
        
        return [{
            "content": doc,
            "score": 1 - dist,  # cosine distance → similarity
            "metadata": meta,
        } for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        )]
    
    def get_stats(self) -> dict:
        return {
            "vi_count": self.collection_vi.count(),
            "en_count": self.collection_en.count(),
        }
```

---

## 2.3 BM25 Keyword Index

#### [NEW] `src/bm25_store.py`

**Keyword search bổ sung cho vector search:**

```python
from rank_bm25 import BM25Okapi
import pickle
import os

class BM25Store:
    def __init__(self):
        self.indices = {"vi": None, "en": None}
        self.documents = {"vi": [], "en": []}
        self.doc_ids = {"vi": [], "en": []}
    
    def build_index(self, chunks: list[dict], language: str):
        """Build BM25 index from chunks"""
        docs = []
        ids = []
        for chunk in chunks:
            # Tokenize: giữ nguyên tên thuốc, split thường
            tokens = self._tokenize(chunk["content"], language)
            docs.append(tokens)
            ids.append(chunk["id"])
        
        self.indices[language] = BM25Okapi(docs)
        self.documents[language] = [c["content"] for c in chunks]
        self.doc_ids[language] = ids
    
    def search(self, query: str, language: str, top_k: int = 5) -> list[dict]:
        """BM25 keyword search"""
        if self.indices[language] is None:
            return []
        
        tokens = self._tokenize(query, language)
        scores = self.indices[language].get_scores(tokens)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        return [{
            "content": self.documents[language][i],
            "score": float(scores[i]),
            "doc_id": self.doc_ids[language][i],
        } for i in top_indices if scores[i] > 0]
    
    def _tokenize(self, text: str, language: str) -> list[str]:
        """Simple tokenization, preserve medical terms"""
        text = text.lower()
        # Keep drug names intact (e.g., warfarin, metformin)
        tokens = text.split()
        return [t.strip(".,;:!?()[]") for t in tokens if len(t) > 1]
    
    def save(self, path: str):
        """Serialize to disk"""
        with open(path, "wb") as f:
            pickle.dump({"indices": self.indices, "documents": self.documents, "doc_ids": self.doc_ids}, f)
    
    def load(self, path: str):
        """Load from disk"""
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.indices = data["indices"]
                self.documents = data["documents"]
                self.doc_ids = data["doc_ids"]
```

---

## 2.4 Hybrid Retriever (RRF Fusion)

#### [NEW] `src/hybrid_retriever.py`

**Kết hợp Vector + BM25 bằng Reciprocal Rank Fusion:**

```python
class HybridRetriever:
    def __init__(self, embedding_manager, vector_store, bm25_store):
        self.embeddings = embedding_manager
        self.vector_store = vector_store
        self.bm25 = bm25_store
    
    def search(self, query: str, language: str = None, top_k: int = 5,
               category_filter: str = None) -> list[dict]:
        """Hybrid search: vector + BM25 → RRF fusion"""
        if language is None:
            language = self.embeddings.detect_language(query)
        
        # 1. Vector search
        query_emb = self.embeddings.embed(query, language)
        vector_results = self.vector_store.search(
            query_emb, language, top_k=top_k * 2, category_filter=category_filter
        )
        
        # 2. BM25 search
        bm25_results = self.bm25.search(query, language, top_k=top_k * 2)
        
        # 3. RRF Fusion
        fused = self._rrf_fusion(vector_results, bm25_results, k=60)
        
        return fused[:top_k]
    
    def _rrf_fusion(self, vector_results, bm25_results, k=60) -> list[dict]:
        """Reciprocal Rank Fusion"""
        scores = {}
        doc_map = {}
        
        # Score from vector search
        for rank, doc in enumerate(vector_results):
            doc_id = doc.get("metadata", {}).get("source", "") + doc["content"][:50]
            scores[doc_id] = scores.get(doc_id, 0) + VECTOR_WEIGHT / (k + rank + 1)
            doc_map[doc_id] = doc
        
        # Score from BM25
        for rank, doc in enumerate(bm25_results):
            doc_id = doc.get("doc_id", "") + doc["content"][:50]
            scores[doc_id] = scores.get(doc_id, 0) + BM25_WEIGHT / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
        
        # Sort by fused score
        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        
        return [{
            **doc_map[doc_id],
            "fused_score": scores[doc_id]
        } for doc_id in sorted_ids]
```

---

## 2.5 Data Ingestion Script

#### [NEW] `main.py` (phần ingest)

```python
def ingest_data():
    """Ingest cleaned data into vector store + BM25"""
    embedding_mgr = DualEmbeddingManager()
    vector_store = VectorStore(CHROMA_PERSIST_DIR)
    bm25_store = BM25Store()
    
    for lang in ["en", "vi"]:
        # Load cleaned chunks
        chunks = load_jsonl(f"data/processed/chunks_{lang}.jsonl")
        
        # Generate embeddings
        texts = [c["content"] for c in chunks]
        embeddings = embedding_mgr.embed_batch(texts, language=lang)
        
        # Store in ChromaDB
        vector_store.add_documents(chunks, embeddings, language=lang)
        
        # Build BM25 index
        bm25_store.build_index(chunks, language=lang)
    
    # Save BM25 to disk
    bm25_store.save("models/bm25_index.pkl")
    
    print(vector_store.get_stats())
```

---

## Verification cho Part 2

```bash
# 1. Test embedding (sẽ download models lần đầu)
python -c "
from src.embeddings import DualEmbeddingManager
emb = DualEmbeddingManager()
vi = emb.embed('Tác dụng phụ của Warfarin', language='vi')
en = emb.embed('Side effects of Warfarin', language='en')
print(f'VI embedding dim: {len(vi)}')
print(f'EN embedding dim: {len(en)}')
print(f'Language detect: {emb.detect_language(\"thuốc warfarin\")}')
"

# 2. Test full ingest
python main.py --ingest

# 3. Test hybrid search
python -c "
from src.hybrid_retriever import HybridRetriever
# ... init components ...
results = retriever.search('Warfarin side effects')
for r in results[:3]:
    print(f'Score: {r[\"fused_score\"]:.4f} | {r[\"content\"][:80]}...')
"
```

**Xong Part 2 → chuyển sang [Part 3](file:///C:/Users/votru/.gemini/antigravity/brain/72dfbf26-8c80-4ebc-85dd-2bf3db7b1dca/plan_part3_safety.md)**
