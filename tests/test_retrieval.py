from src.bm25_store import BM25Store
from src.hybrid_retriever import HybridRetriever
from src.vector_store import VectorStore


class FakeEmbeddingManager:
    def detect_language(self, text):
        return "en"

    def embed(self, text, language=None):
        text = text.lower()
        if "warfarin" in text:
            return [1.0, 0.0, 0.0]
        if "metformin" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def test_hybrid_search_returns_results(tmp_path):
    chunks = [
        {
            "id": "doc1",
            "content": "Warfarin may cause bleeding and has important safety warnings.",
            "category": "drug_safety",
            "source": "test",
            "title": "Warfarin",
        },
        {
            "id": "doc2",
            "content": "Metformin is used for diabetes and may cause stomach side effects.",
            "category": "drug_safety",
            "source": "test",
            "title": "Metformin",
        },
    ]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    vector_store = VectorStore(str(tmp_path / "vectors"), use_chroma=False)
    vector_store.add_documents(chunks, embeddings, "en")
    bm25_store = BM25Store()
    bm25_store.build_index(chunks, "en")
    retriever = HybridRetriever(FakeEmbeddingManager(), vector_store, bm25_store)

    results = retriever.search("Warfarin side effects", "en", top_k=2)

    assert len(results) > 0
    assert results[0]["fused_score"] > 0
    assert "Warfarin" in results[0]["content"]
