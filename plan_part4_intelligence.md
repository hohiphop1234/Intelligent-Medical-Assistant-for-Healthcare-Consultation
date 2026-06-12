# Part 4: Query Router, Response Generation & Validation (~4h)

> Phụ thuộc: Part 1 (config, categories), Part 2 (retrieval)
> Có thể làm song song với Part 3

---

## 4.1 Query Router

#### [NEW] `src/query_router.py`

**LLM-based classifier** — quyết định câu hỏi thuộc category nào, có cần RAG không:

```python
from openai import OpenAI
from dataclasses import dataclass

@dataclass
class QueryClassification:
    intent: str           # "drug_info", "interaction_check", "symptom_inquiry"
    category: str         # "drug_safety", "disease_knowledge", "out_of_scope"
    entities: list[str]   # ["warfarin", "ibuprofen"]
    risk_level: str       # "low", "medium", "high", "critical"
    confidence: float     # 0.0 - 1.0
    requires_rag: bool    # True nếu cần tìm evidence
    language: str         # "en" or "vi"

class QueryRouter:
    def __init__(self):
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
    
    def classify(self, query: str, language: str) -> QueryClassification:
        prompt = f"""Classify this medical query. Return JSON only.

Categories: drug_safety, drug_interaction, overdose_triage, disease_knowledge, 
pregnancy, pediatric, elderly, general_health, out_of_scope

Query: "{query}"

Return:
{{"intent": "...", "category": "...", "entities": [...], "risk_level": "low|medium|high|critical", "confidence": 0.0-1.0, "requires_rag": true|false}}"""

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a medical query classifier. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200,
        )
        
        # Parse JSON response
        result = json.loads(response.choices[0].message.content)
        result["language"] = language
        result["requires_rag"] = True  # FORCE: mọi câu hỏi y tế đều cần RAG
        
        return QueryClassification(**result)
```

**Rules:**
- `confidence < 0.5` → ask clarifying question hoặc refuse
- `requires_rag = True` bắt buộc cho mọi câu hỏi y tế
- `out_of_scope` → bypass RAG, trả lời từ chối ngay

---

## 4.2 Evidence Grader (Corrective RAG)

#### [NEW] `src/evidence_grader.py`

**Grade retrieved chunks** — quyết định evidence có đủ tin cậy không:

```python
@dataclass
class GradingResult:
    relevant_chunks: list[dict]   # Chunks được grade "relevant"
    score: float                   # Average relevance score
    needs_crawl: bool              # True nếu cần crawl thêm
    confidence: str                # "high", "medium", "low"

class EvidenceGrader:
    def __init__(self):
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
    
    def grade(self, question: str, chunks: list[dict]) -> GradingResult:
        relevant = []
        
        for chunk in chunks:
            is_relevant = self._grade_single(question, chunk["content"])
            if is_relevant:
                relevant.append(chunk)
        
        score = len(relevant) / max(len(chunks), 1)
        
        return GradingResult(
            relevant_chunks=relevant,
            score=score,
            needs_crawl=len(relevant) < MIN_EVIDENCE_CHUNKS,
            confidence="high" if score >= 0.6 else "medium" if score >= 0.3 else "low",
        )
    
    def _grade_single(self, question: str, chunk_content: str) -> bool:
        """Grade one chunk: relevant or not"""
        prompt = f"""Given this user question: "{question}"

And this retrieved document:
\"\"\"{chunk_content[:500]}\"\"\"

Is this document relevant and useful for answering the question?
Answer ONLY "relevant" or "irrelevant"."""

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        
        return "relevant" in response.choices[0].message.content.lower()
```

**Decision logic:**
- ≥ 2 chunks "relevant" → đủ evidence, generate response
- < 2 chunks "relevant" → trigger web crawl
- 0 chunks relevant sau crawl → refuse trả lời

---

## 4.3 Web Crawler

#### [NEW] `src/web_crawler.py`

**On-demand crawl từ nguồn tin cậy** (chỉ khi local dataset thiếu):

```python
import requests
from bs4 import BeautifulSoup
import time

class WebCrawler:
    SOURCES = {
        "medlineplus": {
            "search_url": "https://vsearch.nlm.nih.gov/vivisimo/cgi-bin/query-meta?v%3Aproject=medlineplus&v%3Asources=medlineplus-bundle&query={query}",
            "base_url": "https://medlineplus.gov",
        },
        "dailymed": {
            "search_url": "https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={query}",
            "base_url": "https://dailymed.nlm.nih.gov",
        },
    }
    
    def __init__(self):
        self.cache = {}  # URL → content cache
        self.rate_limiter = time.time()
    
    def search(self, query: str, entities: list[str]) -> list[dict]:
        """Search whitelisted sources for additional evidence"""
        results = []
        
        for entity in entities:
            # Try MedlinePlus drug page
            content = self._fetch_medlineplus_drug(entity)
            if content:
                chunks = self._chunk_content(content, source="medlineplus", entity=entity)
                results.extend(chunks)
        
        return results
    
    def _fetch_medlineplus_drug(self, drug_name: str) -> str | None:
        """Fetch drug info from MedlinePlus"""
        url = f"https://medlineplus.gov/druginfo/meds/{drug_name}.html"
        
        if url in self.cache:
            return self.cache[url]
        
        self._rate_limit()
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "MedicalRAG/1.0"})
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                # Extract main content, skip nav/footer
                main = soup.find("article") or soup.find("main") or soup.find("div", {"id": "article"})
                if main:
                    text = main.get_text(separator="\n", strip=True)
                    self.cache[url] = text
                    return text
        except Exception:
            pass
        return None
    
    def _chunk_content(self, content: str, source: str, entity: str) -> list[dict]:
        """Split crawled content into chunks"""
        chunks = []
        words = content.split()
        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_text = " ".join(words[i:i + CHUNK_SIZE])
            if len(chunk_text.split()) >= 30:
                chunks.append({
                    "content": chunk_text,
                    "source": source,
                    "entity": entity,
                    "url": f"https://medlineplus.gov/druginfo/{entity}",
                    "is_crawled": True,
                })
        return chunks
    
    def _rate_limit(self):
        elapsed = time.time() - self.rate_limiter
        if elapsed < 1.0 / CRAWL_RATE_LIMIT:
            time.sleep(1.0 / CRAWL_RATE_LIMIT - elapsed)
        self.rate_limiter = time.time()
```

---

## 4.4 Response Generator

#### [NEW] `src/response_generator.py`

**LLM generate câu trả lời có citations:**

```python
SYSTEM_PROMPT = """You are a medical information assistant. Follow these rules STRICTLY:

1. ONLY answer based on the provided context documents. NEVER use your own knowledge.
2. ALWAYS cite sources using [1], [2], [3] format after each claim.
3. If context is insufficient, say "Không đủ thông tin để trả lời chính xác."
4. NEVER diagnose, prescribe, or recommend specific dosages for individuals.
5. ALWAYS recommend consulting a healthcare professional.
6. For children/pregnancy/elderly: add EXTRA caution warnings.
7. Respond in the SAME LANGUAGE as the user's question.
8. Structure your answer clearly with sections if needed.
9. Mention relevant warnings and side effects from the context."""

class ResponseGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
    
    def generate(self, question: str, chunks: list[dict], 
                 classification: QueryClassification) -> dict:
        # Build context with numbered sources
        context = self._build_context(chunks)
        
        user_prompt = f"""Context documents:
{context}

User question: {question}

Answer the question using ONLY the context above. Cite sources with [1], [2], etc."""

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            stream=True,  # For Chainlit streaming
        )
        
        return {
            "answer_stream": response,  # Stream object for Chainlit
            "sources": self._format_sources(chunks),
            "risk_level": classification.risk_level,
            "category": classification.category,
            "confidence": classification.confidence,
        }
    
    def _build_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("source", chunk.get("source", "Unknown"))
            title = chunk.get("metadata", {}).get("title", "")
            section = chunk.get("metadata", {}).get("section", "")
            parts.append(f"[{i}] Source: {source} | {title} | {section}\n{chunk['content']}\n")
        return "\n---\n".join(parts)
    
    def _format_sources(self, chunks: list[dict]) -> list[dict]:
        return [{
            "index": i + 1,
            "title": c.get("metadata", {}).get("title", c.get("source", "Unknown")),
            "source": c.get("metadata", {}).get("source", ""),
            "url": c.get("metadata", {}).get("url", ""),
            "section": c.get("metadata", {}).get("section", ""),
            "score": c.get("fused_score", c.get("score", 0)),
        } for i, c in enumerate(chunks)]
```

---

## 4.5 Response Validator

#### [NEW] `src/response_validator.py`

**Post-generation safety checks:**

```python
class ResponseValidator:
    PROHIBITED_PATTERNS = [
        r"you should take \d+ ?mg",          # Specific dosage recommendation
        r"I diagnose you with",               # Diagnosis
        r"I prescribe",                        # Prescription
        r"this will cure",                     # Cure claims
        r"tôi chẩn đoán",                     # VN diagnosis
        r"bạn nên uống \d+",                  # VN specific dosage
        r"thuốc này sẽ chữa",                 # VN cure claims
    ]
    
    def validate(self, response: dict, chunks: list[dict]) -> dict:
        answer = response.get("answer", "")
        issues = []
        
        # 1. Citation check
        cited_nums = re.findall(r'\[(\d+)\]', answer)
        for num in cited_nums:
            if int(num) > len(chunks):
                issues.append(f"Citation [{num}] references non-existent source")
        
        # 2. Prohibited content check
        for pattern in self.PROHIBITED_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                issues.append(f"Prohibited pattern found: {pattern}")
        
        # 3. No-citation check (answer has claims but no citations)
        if len(answer) > 100 and not cited_nums:
            issues.append("Answer contains claims but no citations")
        
        # 4. Add disclaimer based on risk level
        disclaimer = get_disclaimer(response["risk_level"], response.get("language", "vi"))
        
        response["disclaimer"] = disclaimer
        response["validation_issues"] = issues
        response["is_valid"] = len(issues) == 0
        
        return response
```

---

## Verification cho Part 4

```bash
# Test query router
python -c "
from src.query_router import QueryRouter
qr = QueryRouter()

r1 = qr.classify('Tác dụng phụ của Warfarin?', 'vi')
print(f'Category: {r1.category}, Risk: {r1.risk_level}, Entities: {r1.entities}')

r2 = qr.classify('Can I take ibuprofen with warfarin?', 'en')
print(f'Category: {r2.category}, Risk: {r2.risk_level}, Entities: {r2.entities}')

r3 = qr.classify('Thời tiết hôm nay thế nào?', 'vi')
print(f'Category: {r3.category} (should be out_of_scope)')
"

# Test evidence grader
python -c "
from src.evidence_grader import EvidenceGrader
eg = EvidenceGrader()
# ... test with sample chunks ...
"
```

**Xong Part 4 → chuyển sang [Part 5](file:///C:/Users/votru/.gemini/antigravity/brain/72dfbf26-8c80-4ebc-85dd-2bf3db7b1dca/plan_part5_pipeline_ui.md)**
