# LegalMind RAG Testing & Evaluation

This directory contains the comprehensive testing and evaluation framework for the LegalMind RAG system, implementing industry best practices for AI-powered legal research.

## 🏛️ Overview

The evaluation framework implements the **"RAG Triad"** metrics and automated testing patterns required for production legal AI systems:

1. **Faithfulness (Groundedness)** - Prevents hallucinations by ensuring answers derive only from source documents
2. **Answer Relevance** - Validates that responses actually address the user's question
3. **Context Precision** - Ensures the most relevant information ranks highest in search results

## 🔬 Testing Architecture

### Evaluation Agents

**1. Adversarial Lawyer Agent** (Synthetic Test Generator)
- Generates complex, multi-hop legal questions from your documents
- Creates ground-truth (Question, Context, Answer) triples for benchmarking
- Produces 50+ test cases automatically for "Golden Dataset" creation

**2. Compliance Auditor Agent** (Hallucination Detector)  
- Performs "LLM-as-a-judge" fact-checking using DeepEval/Ragas
- Extracts claims from responses and cross-references against source chunks
- Calculates faithfulness scores and flags hallucinations

**3. Shepardizer Agent** (Citation Validator)
- Validates every citation reference exists in source documents
- Ensures proper source attribution and link accuracy
- Critical for legal compliance and auditability standards

### CI/CD Quality Gates

- **Faithfulness Threshold**: ≥ 0.9 (90% accuracy requirement)
- **Answer Relevance**: ≥ 0.8 (80% relevance requirement)
- **Context Precision**: ≥ 0.85 (85% precision requirement)
- **Citation Accuracy**: ≥ 95% (95% valid citations)

## 🚀 Quick Start

### Run Unit Tests
```bash
# Run all tests
pytest tests/test_evaluation.py -v

# Run specific test categories
pytest tests/test_evaluation.py::TestIngestionPipeline -v
pytest tests/test_evaluation.py::TestRAGMetrics -v
pytest tests/test_evaluation.py::TestShepardizerAgent -v
```

### Generate Golden Dataset
```bash
# Generate synthetic test data from your documents
python run_evaluation.py --generate-golden --docs-path data/docs --n-questions 50
```

### Run Full Evaluation
```bash
# Complete evaluation pipeline
python run_evaluation.py --full

# Just run metrics on existing dataset
python run_evaluation.py --run-metrics
```

### GitHub Actions CI/CD

The evaluation suite runs automatically on every Pull Request:

```yaml
name: LegalMind RAG Evaluation
on:
  pull_request:
    branches: [main]

jobs:
  rag-evaluation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run RAG Evaluation Suite
        run: pytest tests/test_evaluation.py::TestRAGMetrics -v
```

**Build fails if faithfulness drops below 0.9** - ensuring no regression in response quality.

## 📊 Evaluation Metrics

### Faithfulness (Groundedness)
```python
def test_faithfulness_above_threshold(self, sample_eval_sample, grounded_rag_response):
    auditor = ComplianceAuditorAgent()
    result = auditor.evaluate(sample_eval_sample, grounded_rag_response)
    
    assert result["faithfulness"] >= FAITHFULNESS_THRESHOLD, (
        f"FAITHFULNESS REGRESSION: {result['faithfulness']:.2f} < {FAITHFULNESS_THRESHOLD}. "
        "This build is blocked."
    )
```

### Answer Relevance
```python  
def test_answer_relevance_scored(self, sample_eval_sample, rag_response):
    auditor = ComplianceAuditorAgent()
    result = auditor.evaluate(sample_eval_sample, rag_response)
    assert result["answer_relevance"] >= 0.8
```

### Context Precision
```python
def test_context_precision_scored(self, sample_eval_sample, grounded_rag_response):
    agent = ShepardizerAgent()
    result = agent.validate(sample_eval_sample.question, grounded_rag_response)
    assert result["context_precision"] >= 0.85
```

## 🏗️ Synthetic Test Generation

The Adversarial Lawyer Agent automatically generates test cases:

```python
samples = adversarial_agent.generate_golden_dataset(legal_chunks, n_questions=50)

# Creates structured test data:
# {
#   "question": "How does Clause X interact with liability limits in Contract B?",
#   "reference_context": "exact quote from the document that answers this",
#   "expected_answer": "The liability is limited to...",
#   "doc_ids": ["doc-abc123"]
# }
```

## 📈 Production Evaluation Report

```
============================================================
🏛️  LEGALMIND RAG EVALUATION REPORT
============================================================
📊 METRICS SUMMARY
   Total Samples: 50
   Faithfulness: 0.942 (threshold: 0.9)
   Answer Relevance: 0.876 (threshold: 0.8) 
   Context Precision: 0.891 (threshold: 0.85)
   Citation Accuracy: 97.2%

🎯 QUALITY GATES
   Faithfulness (No Hallucinations): ✅ PASS
   Answer Relevance: ✅ PASS
   Context Precision: ✅ PASS
   Citation Validation: ✅ PASS

🏆 OVERALL: ✅ EVALUATION PASSED
============================================================
```

## 🔧 Configuration

Configure thresholds in `core/config.py`:

```python
@dataclass
class Settings:
    faithfulness_threshold: float = 0.9
    relevance_threshold: float = 0.8
    precision_threshold: float = 0.85
```

## 🎯 Industry Best Practices

✅ **Shift-Left Testing** - Evaluation runs in development, not just production  
✅ **LLM-as-a-Judge** - Automated evaluation using GPT-4o for consistency  
✅ **Golden Dataset** - Synthetic test data generation from your own documents  
✅ **Quality Gates** - Build fails if metrics regress below thresholds  
✅ **Source Attribution** - Every claim must cite source documents  
✅ **Hallucination Detection** - Explicit faithfulness scoring prevents false claims  

## 📝 Integration with DeepEval/Ragas

The agents are compatible with industry-standard evaluation frameworks:

```python
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# Can integrate with existing DeepEval workflows
metric = FaithfulnessMetric(threshold=0.9)
test_case = LLMTestCase(input=question, actual_output=answer, retrieval_context=chunks)
metric.measure(test_case)
```

This comprehensive testing framework ensures your legal RAG system meets the highest standards for accuracy, reliability, and compliance.