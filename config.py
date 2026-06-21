from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv:
    load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# LLM via OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen-plus")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# Local LLM
LORA_CHECKPOINT_DIR = str(BASE_DIR / "ckpt" / "qwen3_4b_thinking_meddies_lora")

# Embeddings
EMBEDDING_MODEL_VI = os.getenv(
    "EMBEDDING_MODEL_VI", "Dqdung205/medical_vietnamese_embedding"
)
FALLBACK_EMBEDDING_DIM = int(os.getenv("FALLBACK_EMBEDDING_DIM", "384"))
FORCE_FALLBACK_EMBEDDINGS = (
    os.getenv("FORCE_FALLBACK_EMBEDDINGS", "false").lower() == "true"
)

# Vector store
CHROMA_PERSIST_DIR = str(BASE_DIR / "models" / "chromadb")
COLLECTION_NAME_VI = "medical_rag_vi"

# Retrieval
TOP_K = int(os.getenv("TOP_K", "8"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.6"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
EVIDENCE_THRESHOLD = float(os.getenv("EVIDENCE_THRESHOLD", "0.5"))
MIN_EVIDENCE_CHUNKS = int(os.getenv("MIN_EVIDENCE_CHUNKS", "2"))

# Data paths
RAW_DATA_DIR = str(BASE_DIR / "data" / "rag_processed")
PROCESSED_DATA_DIR = str(BASE_DIR / "data" / "processed")
CATEGORIES_PATH = str(BASE_DIR / "data" / "categories.json")
RAW_DATA_FALLBACK_DIR = str(Path.home() / "Downloads" / "rag_processed")
RAW_DATA_FALLBACK_DIRS = [
    candidate
    for candidate in [
        os.getenv("RAG_DATA_SOURCE_DIR"),
        os.getenv("RAG_DATA_DIR"),
        str(BASE_DIR / "data" / "rag_processed"),
        str(Path.home() / "Downloads" / "data" / "rag_processed"),
        RAW_DATA_FALLBACK_DIR,
    ]
    if candidate
]

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Safety
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
EMERGENCY_RESPONSE_ONLY = os.getenv("EMERGENCY_RESPONSE_ONLY", "true").lower() == "true"

# Crawl
CRAWL_WHITELIST = [
    "medlineplus.gov",
    "dailymed.nlm.nih.gov",
    "fda.gov",
    "who.int",
    "cdc.gov",
]
CRAWL_CACHE_DIR = str(BASE_DIR / "data" / "crawl_cache")
CRAWL_RATE_LIMIT = float(os.getenv("CRAWL_RATE_LIMIT", "2"))
