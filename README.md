# LegalMind RAG

Production-grade legal research assistant using modular RAG architecture with hybrid retrieval, semantic caching, and automated quality evaluation.

## 🏛️ High-Level Design

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │ -> │  LegalMindRAG    │ -> │ Redis Cache     │
└─────────────────┘    │    Pipeline      │    └─────────────────┘
                       └──────────────────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
            ┌───────▼───┐ ┌───▼───┐ ┌───▼────────┐
            │ Semantic  │ │ BM25  │ │ Cohere     │
            │ Vector    │ │ Search│ │ Rerank     │
            └───────────┘ └───────┘ └────────────┘
                    │         │         │
                    └─────────┼─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Qdrant Vector DB  │
                    └───────────────────┘
```

**Core Components:**
- **Ingestion**: Semantic chunking by legal document structure (ARTICLE/Section boundaries)
- **Retrieval**: Hybrid Vector (Qdrant) + BM25 with Reciprocal Rank Fusion
- **Re-ranking**: Cohere rerank-v4.0 (top 20 → top 5)
- **Generation**: GPT-4o with mandatory citations and hallucination prevention
- **Caching**: Redis semantic cache with embedding similarity

## 🚀 Quick Start

### Setup
```bash
# Clone and setup
git clone https://github.com/rashad-leapai/legal-mind.git
cd legal-mind
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Environment variables
cp .env.example .env
# Add your API keys to .env
```

### Run Application
```bash
streamlit run app.py
```

### Run Evaluation
```bash
# Unit tests (fast)
pytest tests/test_evaluation.py -v

# Production evaluation (comprehensive)
python run_evaluation.py --full
```

## 📁 File Structure

```
legal-mind/
├── app.py                    # Streamlit UI
├── core/
│   ├── pipeline.py          # Main RAG orchestrator
│   ├── ingestion.py         # Semantic chunking
│   ├── retriever.py         # Hybrid retrieval + RRF
│   ├── generation.py        # GPT-4o with citations
│   ├── cache.py             # Redis semantic cache
│   ├── vector_store.py      # Qdrant integration
│   └── models.py            # Data models
├── agents/                  # Evaluation agents
│   ├── adversarial_lawyer.py    # Test data generation
│   ├── compliance_auditor.py    # Hallucination detection
│   └── shepardizer.py           # Citation validation
├── tests/
│   └── test_evaluation.py       # RAG quality tests
├── data/sample_docs/            # Legal documents
├── .github/workflows/           # CI/CD evaluation
└── run_evaluation.py            # Production evaluation script
```

## 🧪 Quality Metrics

**Automated CI/CD Thresholds:**
- **Faithfulness**: ≥ 0.9 (no hallucinations)
- **Answer Relevance**: ≥ 0.8 (addresses question)
- **Context Precision**: ≥ 0.85 (relevant chunks ranked high)
- **Citation Accuracy**: 100% (valid source references)

## 📊 Features

✅ **Modular Architecture** - Swap components without breaking system  
✅ **Semantic Chunking** - Legal structure boundaries (not fixed sizes)  
✅ **Hybrid Retrieval** - Vector + BM25 with reranking  
✅ **Source Attribution** - Mandatory [DOC:id] citations  
✅ **Hallucination Prevention** - Strict context-only responses  
✅ **Semantic Caching** - Redis-based query optimization  
✅ **Automated Testing** - CI/CD quality gates on every PR
