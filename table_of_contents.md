# Table of Contents (Streamlined — 30-50 pages)

## Intelligent Medical Assistant for Healthcare Consultation Using Retrieval-Augmented Generation (RAG)

---

> [!NOTE]
> **Annotation Legend:**
> 📐 Diagram &nbsp;|&nbsp; 📊 Chart &nbsp;|&nbsp; 💻 Code &nbsp;|&nbsp; 📋 Table &nbsp;|&nbsp; 📸 Screenshot
>
> ⭐ = Critical item (must-have for presentation/defense)

---

**List of Figures**
**List of Tables**
**List of Abbreviations**
**Abstract**

---

### Chapter 1: Introduction (~3-4 pages)
- 1.1 Problem Statement and Motivation
- 1.2 Research Objectives
- 1.3 Scope and Limitations
- 1.4 Report Structure Overview

> *Text-only. No figures needed.*

---

### Chapter 2: Theoretical Background and Related Work (~6-8 pages)
- 2.1 Large Language Models and Fine-Tuning
  - 2.1.1 Overview of LLMs (Qwen3 Family)
    - 📋 **Table 2.1** — Comparison of LLM families (GPT, LLaMA, Qwen) — why Qwen3-4B was chosen
  - 2.1.2 LoRA and Parameter-Efficient Fine-Tuning
    - 📐 **Fig 2.1** — ⭐ LoRA architecture diagram (frozen weights + low-rank A, B matrices)

- 2.2 Retrieval-Augmented Generation (RAG)
  - 2.2.1 RAG Fundamentals
    - 📐 **Fig 2.2** — ⭐ Basic RAG pipeline (Query → Retrieve → Augment → Generate)
  - 2.2.2 Naive RAG vs. Advanced RAG vs. Agentic RAG
    - 📋 **Table 2.2** — Comparison table of 3 RAG paradigms

- 2.3 Information Retrieval: Vector Search, BM25, and Hybrid Fusion
  - 📐 **Fig 2.3** — Hybrid retrieval concept (Vector + BM25 → RRF fusion)

- 2.4 Safety in Medical AI Systems

- 2.5 Related Work
  - 📋 **Table 2.3** — Comparison with existing medical QA systems

---

### Chapter 3: System Design and Architecture (~8-10 pages)
- 3.1 System Overview
  - 📐 **Fig 3.1** — ⭐ **Full system architecture diagram** (Frontend ↔ FastAPI ↔ LangGraph ↔ ChromaDB/BM25 ↔ Qwen3-4B)
  - 📋 **Table 3.1** — Technology stack summary

- 3.2 Agentic RAG Pipeline (LangGraph)
  - 📐 **Fig 3.2** — ⭐ **LangGraph pipeline flowchart** (START → safety_check → classifier → [general_qa | medical_rag] → END)
  - 💻 **Code 3.1** — `MedicalState` TypedDict definition

- 3.3 Safety and Risk Management
  - 📐 **Fig 3.3** — Safety check decision tree
  - 📋 **Table 3.2** — Risk levels (Low/Medium/High/Critical) with examples and disclaimers
  - 💻 **Code 3.2** — Emergency detection regex patterns (snippet)

- 3.4 Hybrid Retrieval System
  - 📐 **Fig 3.4** — ⭐ **Hybrid retrieval fusion diagram** (Vector 0.6 + BM25 0.4 → RRF → ranked results)
  - 💻 **Code 3.3** — RRF implementation (key function)
  - 📋 **Table 3.3** — Retrieval parameters (TOP_K, weights, chunk_size, overlap)

- 3.5 Evidence Grading and Web Crawl Fallback
  - 📐 **Fig 3.5** — Evidence grading flow (grade → sufficient? → generate / crawl fallback)
  - 📋 **Table 3.4** — Trusted source whitelist (WHO, CDC, FDA, MedlinePlus, DailyMed)

- 3.6 Response Generation and Validation
  - 💻 **Code 3.4** — System prompt template (citation rules, no diagnosis/prescription)

- 3.7 Frontend and API Design
  - 📋 **Table 3.5** — API endpoints (POST /api/chat, POST /api/chat/stream, GET /api/stats)
  - 📸 **Fig 3.6** — ⭐ **Chat UI screenshot** (showing conversation with citations + disclaimer)

---

### Chapter 4: Data and Model Preparation (~6-8 pages)
- 4.1 Data Sources and Collection
  - 📋 **Table 4.1** — Data sources overview (name, type, size, language)
  - 📊 **Fig 4.1** — Category distribution chart (7 medical categories)

- 4.2 Data Preprocessing Pipeline
  - 📐 **Fig 4.2** — ⭐ Preprocessing pipeline (Raw → clean → translate → chunk → annotate → ingest)
  - 📋 **Table 4.2** — Chunking parameters (size=800, overlap=150) and filtering criteria

- 4.3 Knowledge Base Construction
  - 📋 **Table 4.3** — ⭐ Final dataset statistics:

    | Asset | Count | Size |
    |-------|-------|------|
    | Processed chunks | ~19,260 | ~78 MB |
    | ChromaDB vectors | ~19,260 | ~859 MB |
    | BM25 index | ~19,260 | ~187 MB |

- 4.4 LLM Fine-Tuning
  - 4.4.1 Risk-Stratified Training Strategy
    - 📋 **Table 4.4** — ⭐ Category inclusion/exclusion (General QA ✅ trained, High-risk categories ❌ RAG-only)
  - 4.4.2 LoRA Configuration and Training
    - 📋 **Table 4.5** — LoRA + training hyperparameters (rank, alpha, epochs, batch_size, learning_rate)
    - 💻 **Code 4.1** — SFTTrainer config snippet
    - 📊 **Fig 4.3** — ⭐ Training loss curve
  - 4.4.3 DDP Training on Kaggle (2× T4 GPUs)
    - 📋 **Table 4.6** — Training environment and results (GPU, VRAM, time, final loss)

- 4.5 Embedding Model
  - 📋 **Table 4.7** — Embedding specs (`Dqdung205/medical_vietnamese_embedding`, dim=384)

---

### Chapter 5: Implementation (~5-7 pages)
- 5.1 Project Structure
  - 📐 **Fig 5.1** — Project directory tree (backend/src/, frontend/src/, notebooks/)

- 5.2 Backend Core Modules
  - 📋 **Table 5.1** — ⭐ Module summary table:

    | Module | Purpose |
    |--------|---------|
    | `langgraph_pipeline.py` | LangGraph workflow orchestrator |
    | `hybrid_retriever.py` | Vector + BM25 fusion |
    | `safety_guard.py` | Emergency detection + scope filter |
    | `evidence_grader.py` | Evidence quality scoring |
    | `response_generator.py` | Cited answer generation |
    | `qwen_llm.py` | Qwen3-4B via Ollama (Singleton) |
    | `web_crawler.py` | Fallback evidence crawling |
    | ... | ... |

  - 💻 **Code 5.1** — ⭐ LangGraph StateGraph construction (add_node, add_conditional_edges, compile)

- 5.3 Frontend Implementation
  - 📸 **Fig 5.2** — ⭐ Chat UI screenshots (idle + active conversation)
  - 📸 **Fig 5.3** — Responsive layout (mobile vs. desktop) *(optional)*

- 5.4 Deployment
  - 💻 **Code 5.2** — Ollama Modelfile content
  - 📋 **Table 5.2** — Hardware requirements (CUDA GPU, ~4GB VRAM)

---

### Chapter 6: Evaluation and Conclusion (~5-7 pages)
- 6.1 Evaluation Framework
  - 📋 **Table 6.1** — Test suite composition (60 medical + 15 emergency + 15 out-of-scope + 10 general QA)

- 6.2 Results
  - 📋 **Table 6.2** — ⭐ **Main results table:**

    | Metric | Score |
    |--------|-------|
    | Emergency Detection | 100% |
    | Out-of-scope Refusal | 90% |
    | Citation Accuracy | 93% |
    | Disclaimer Presence | 100% |
    | Prohibited Content | 0% |

  - 📊 **Fig 6.1** — Bar chart of evaluation metrics
  - 📊 **Fig 6.2** — Retrieval comparison (Vector-only vs. BM25-only vs. Hybrid)
  - 📸 **Fig 6.3** — Sample correct/incorrect response screenshots

- 6.3 Discussion
  - 6.3.1 Strengths
  - 6.3.2 Limitations

- 6.4 Conclusion

- 6.5 Future Work

---

**References**

**Appendices**
- Appendix A: Full System Configuration (`config.py`)
- Appendix B: Sample Queries and Responses (3-5 examples with screenshots)
- Appendix C: API Documentation

---

## Visual Elements Summary

| Type | Count |
|------|-------|
| 📐 Architecture / Flow Diagrams | ~10 |
| 📊 Charts | ~4 |
| 💻 Code Snippets | ~8 |
| 📋 Tables | ~16 |
| 📸 Screenshots | ~4 |
| **Total** | **~42** |

> [!IMPORTANT]
> **10 must-have ⭐ items for defense:**
> 1. Fig 2.1 — LoRA architecture
> 2. Fig 2.2 — RAG pipeline concept
> 3. Fig 3.1 — Full system architecture
> 4. Fig 3.2 — LangGraph pipeline flowchart
> 5. Fig 3.4 — Hybrid retrieval fusion
> 6. Fig 3.6 — Chat UI screenshot
> 7. Table 4.4 — Risk-stratified training strategy
> 8. Fig 4.3 — Training loss curve
> 9. Code 5.1 — LangGraph StateGraph code
> 10. Table 6.2 — Evaluation results
