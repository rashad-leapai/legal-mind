# LegalMind — AI Legal Research Assistant

A production-grade RAG system for querying 10,000+ legal case files and contracts with source citations and zero hallucination tolerance.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     LegalMindRAG Pipeline                        │
│                                                                  │
│  ┌──────────────┐    ┌────────────────┐    ┌─────────────────┐  │
│  │  Ingestion   │    │   Retrieval    │    │   Generation    │  │
│  │              │    │                │    │                 │  │
│  │ PDF/OCR      │    │ QdrantDB       │    │ GPT-4o          │  │
│  │ Recursive    │───▶│ (vector)   +   │───▶│ Temperature=0   │  │
│  │ Chunking     │    │ BM25           │    │ Citation-forced │  │
│  │ 512t/10%ovlp │    │ ──▶ RRF Merge  │    │ system prompt   │  │
│  │ Metadata     │    │ ──▶ Cohere     │    │                 │  │
│  │ Enrichment   │    │     Rerank     │    │                 │  │
│  └──────────────┘    └────────────────┘    └─────────────────┘  │
│                                │                                 │
│                    ┌───────────▼──────────┐                     │
│                    │   Semantic Cache      │                     │
│                    │   (Redis)            │                     │
│                    └──────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        Three Agents                              │
│                                                                  │
│  🤺 AdversarialLawyer  → Generates golden test dataset (50+ Q&A)│
│  🔍 ComplianceAuditor  → Faithfulness + Hallucination detection  │
│  📎 Shepardizer        → Citation validation + Context precision │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start (Cloud-First)

This project works out-of-the-box with cloud services. No Docker required!

```bash
git clone <your-repo-url>
cd legal-mind
python quick_setup.py
```

Then edit `.env` with your API keys and run:
```bash
streamlit run app.py
```

## 🔑 Required API Keys

1. **OpenAI API** - Get free credits at [platform.openai.com](https://platform.openai.com/api-keys)
2. **Cohere API** - Free tier at [dashboard.cohere.ai](https://dashboard.cohere.ai/api-keys)
3. **Qdrant Cloud** (Optional) - Free tier at [cloud.qdrant.io](https://cloud.qdrant.io/)
4. **Redis Cloud** (Optional) - Free tier at [redis.com/try-free](https://redis.com/try-free/)

*Note: Qdrant and Redis have demo instances configured by default for testing*

## 🐳 Alternative: Local Development

If you prefer local development with Docker:

```bash
# Start local infrastructure
docker-compose up -d qdrant redis

# Set local mode in .env
USE_LOCAL_QDRANT=true
USE_LOCAL_REDIS=true

# Run the app
streamlit run app.py
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Modular component injection | Swap QdrantDB for Pinecone, or GPT-4o for Claude without touching orchestration |
| Reciprocal Rank Fusion | Merges vector + BM25 without requiring score normalization |
| Cross-encoder reranking (Cohere) | Pushes semantically close but contextually weak chunks down |
| Temperature=0 on generation | Legal accuracy requires deterministic outputs |
| Redis semantic cache | Eliminates repeat LLM calls for similar questions — cuts cost 40-60% in practice |
| `[DOC:id]` inline citation format | Machine-parseable by the Shepardizer agent for automated validation |

## CI/CD Quality Gates

GitHub Actions runs on every PR:
- **Faithfulness ≥ 0.9** → blocks merge if violated  
- **Citation validation** → fails if broken `[DOC:id]` references detected
- **Context Precision** → scored and reported per PR

## Project Structure

```
legalmind/
├── app.py                  # Streamlit frontend
├── core/
│   ├── config.py           # Settings (pydantic)
│   ├── models.py           # Data models
│   ├── ingestion.py        # PDF/text parsing + chunking
│   ├── vector_store.py     # ChromaDB abstraction
│   ├── bm25_retriever.py   # Keyword search
│   ├── retriever.py        # Hybrid retrieval + Cohere rerank
│   ├── cache.py            # Redis semantic cache
│   ├── generation.py       # LLM generation with citation prompt
│   └── pipeline.py         # Orchestrator
├── agents/
│   ├── adversarial_lawyer.py   # Golden dataset generator
│   ├── compliance_auditor.py   # Hallucination detector (LLM-as-judge)
│   └── shepardizer.py          # Citation + precision validator
├── tests/
│   └── test_evaluation.py  # Pytest suite with mocked evals
└── .github/workflows/
    └── rag_eval.yml        # CI/CD pipeline
```
