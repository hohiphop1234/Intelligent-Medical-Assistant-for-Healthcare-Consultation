# Part 1: Project Setup & Data Preparation (~2h)

## 1.1 Dependencies

#### [NEW] `requirements.txt`

```
openai>=1.30.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
transformers>=4.40.0
torch>=2.0.0
chainlit>=1.1.0
rank-bm25>=0.2.2
ftfy>=6.0
langdetect>=1.0.9
ragas>=0.1.0
requests>=2.31.0
beautifulsoup4>=4.12.0
python-dotenv>=1.0.0
numpy>=1.24.0
```

#### [NEW] `.env`

```
OPENROUTER_API_KEY=your_key_here
```

---

## 1.2 Configuration

#### [NEW] `config.py`

```python
import os
from dotenv import load_dotenv
load_dotenv()

# === LLM (Qwen 3.6 via OpenRouter) ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "qwen/qwen3.6-plus"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048

# === Embedding (Dual Model) ===
EMBEDDING_MODEL_VI = "Dqdung205/medical_vietnamese_embedding"
EMBEDDING_MODEL_EN = "BAAI/bge-m3"

# === Vector Store ===
CHROMA_PERSIST_DIR = "models/chromadb"
COLLECTION_NAME_VI = "medical_rag_vi"
COLLECTION_NAME_EN = "medical_rag_en"

# === Retrieval ===
TOP_K = 5
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4
EVIDENCE_THRESHOLD = 0.5       # min score để chấp nhận evidence
MIN_EVIDENCE_CHUNKS = 2        # cần ít nhất 2 chunks relevant

# === Data Paths ===
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
CATEGORIES_PATH = "data/categories.json"

# === Chunking ===
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# === Safety ===
CONFIDENCE_THRESHOLD = 0.5     # dưới mức này → refuse
EMERGENCY_RESPONSE_ONLY = True  # emergency → không qua RAG

# === Crawl ===
CRAWL_WHITELIST = [
    "medlineplus.gov",
    "dailymed.nlm.nih.gov",
    "fda.gov",
    "who.int",
    "cdc.gov",
]
CRAWL_CACHE_DIR = "data/crawl_cache"
CRAWL_RATE_LIMIT = 2  # requests/second
```

---

## 1.3 Data Cleaning

#### [NEW] `src/data_cleaner.py`

Xử lý dataset từ `rag_processed`:

**Chức năng chính:**
- **Fix encoding tiếng Việt**: Dùng `ftfy` library để sửa mojibake (`chÃnh` → `chính`)
- **Remove noise**: Regex patterns loại bỏ navigation, breadcrumbs, copyright, menu items
- **Normalize whitespace**: Collapse multiple newlines, trim trailing spaces
- **Validate quality**: Reject chunks < 50 words hoặc > 80% special characters
- **Enrich metadata**: Thêm `language`, `word_count`, `quality_score` vào mỗi chunk

**Input:** `rag_chunks.jsonl` + `rag_chunks_vi.jsonl` từ `Downloads/rag_processed`
**Output:** `data/processed/chunks_en.jsonl` + `data/processed/chunks_vi.jsonl`

```python
class DataCleaner:
    def fix_vietnamese_encoding(self, text: str) -> str:
        """Fix mojibake using ftfy"""
        return ftfy.fix_text(text)
    
    def remove_noise(self, text: str) -> str:
        """Remove menu items, headers, footers, breadcrumbs"""
        patterns = [
            r"Skip to main content",
            r"U\.S\. National Library of Medicine",
            r"MedlinePlus.*Trusted Health Information",
            r"Page last updated:.*",
            r"URL of this page:.*",
        ]
        for p in patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE)
        return text.strip()
    
    def validate_chunk(self, chunk: dict) -> bool:
        """Reject low-quality chunks"""
        words = chunk["content"].split()
        if len(words) < 50:
            return False
        special_ratio = sum(1 for c in chunk["content"] if not c.isalnum() and not c.isspace()) / len(chunk["content"])
        if special_ratio > 0.3:
            return False
        return True
    
    def process_dataset(self, raw_dir: str, output_dir: str):
        """Full pipeline: load → clean → validate → save"""
        # Process English chunks
        # Process Vietnamese chunks (with encoding fix)
        # Save cleaned versions
```

---

## 1.4 Category Definitions

#### [NEW] `data/categories.json`

```json
{
  "categories": [
    {
      "id": "drug_safety",
      "name": "Drug Safety & Information",
      "name_vi": "An toàn thuốc & Thông tin thuốc",
      "keywords_en": ["drug", "medication", "medicine", "dose", "side effect", "adverse"],
      "keywords_vi": ["thuốc", "liều", "tác dụng phụ", "phản ứng"],
      "risk_level": "high"
    },
    {
      "id": "drug_interaction",
      "name": "Drug Interactions",
      "name_vi": "Tương tác thuốc",
      "keywords_en": ["interaction", "combine", "together with", "mixing"],
      "keywords_vi": ["tương tác", "kết hợp", "dùng chung"],
      "risk_level": "critical"
    },
    {
      "id": "overdose_triage",
      "name": "Overdose & Emergency Triage",
      "name_vi": "Quá liều & Cấp cứu",
      "keywords_en": ["overdose", "too much", "poisoning", "emergency"],
      "keywords_vi": ["quá liều", "uống nhiều", "ngộ độc", "cấp cứu"],
      "risk_level": "critical"
    },
    {
      "id": "disease_knowledge",
      "name": "Disease & Condition Information",
      "name_vi": "Thông tin bệnh lý",
      "keywords_en": ["disease", "condition", "symptoms", "treatment", "cause"],
      "keywords_vi": ["bệnh", "triệu chứng", "điều trị", "nguyên nhân"],
      "risk_level": "medium"
    },
    {
      "id": "pregnancy",
      "name": "Pregnancy & Breastfeeding",
      "name_vi": "Thai kỳ & Cho con bú",
      "keywords_en": ["pregnancy", "pregnant", "breastfeeding", "fetal"],
      "keywords_vi": ["thai kỳ", "mang thai", "cho con bú", "thai nhi"],
      "risk_level": "critical"
    },
    {
      "id": "pediatric",
      "name": "Pediatric (Children)",
      "name_vi": "Nhi khoa (Trẻ em)",
      "keywords_en": ["child", "children", "pediatric", "infant", "baby"],
      "keywords_vi": ["trẻ em", "trẻ nhỏ", "nhi khoa", "trẻ sơ sinh"],
      "risk_level": "critical"
    },
    {
      "id": "elderly",
      "name": "Elderly & Geriatric",
      "name_vi": "Người cao tuổi",
      "keywords_en": ["elderly", "geriatric", "older adults", "aging"],
      "keywords_vi": ["người già", "người cao tuổi", "lão khoa"],
      "risk_level": "high"
    }
  ],
  "out_of_scope": {
    "description": "Questions not related to medical/health topics",
    "examples_en": ["How to cook pasta?", "What's the weather?", "Help me code"],
    "examples_vi": ["Nấu mì ý thế nào?", "Thời tiết hôm nay?", "Giúp mình code"]
  }
}
```

---

## Verification cho Part 1

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify data cleaning
python -c "from src.data_cleaner import DataCleaner; dc = DataCleaner(); dc.process_dataset('data/raw', 'data/processed')"

# 3. Check output
python -c "
import json
with open('data/processed/chunks_en.jsonl') as f:
    chunks = [json.loads(l) for l in f]
print(f'EN chunks: {len(chunks)}')
with open('data/processed/chunks_vi.jsonl') as f:
    chunks = [json.loads(l) for l in f]
print(f'VI chunks: {len(chunks)}')
print('Sample:', chunks[0]['content'][:100])
"
```

**Xong Part 1 → chuyển sang [Part 2](file:///C:/Users/votru/.gemini/antigravity/brain/72dfbf26-8c80-4ebc-85dd-2bf3db7b1dca/plan_part2_retrieval.md)**
