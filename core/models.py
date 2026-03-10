"""Data models for the LegalMind RAG system."""
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a legal document."""
    chunk_id: str
    doc_id: str
    content: str
    metadata: dict[str, Any]
    embedding: list[float] | None = None
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = str(uuid.uuid4())


@dataclass
class RetrievedChunk:
    """Represents a chunk retrieved during RAG search."""
    chunk: DocumentChunk
    score: float
    retrieval_method: str  # 'vector', 'bm25', 'hybrid', 'reranked'


@dataclass
class RAGResponse:
    """Response from the RAG system."""
    answer: str
    source_ids: list[str]
    retrieved_chunks: list[RetrievedChunk]
    cached: bool = False
    
    
@dataclass
class EvaluationSample:
    """Sample for evaluation by agents."""
    question: str
    reference_context: str
    expected_answer: str
    doc_ids: list[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Result from agent evaluation."""
    faithfulness: float
    answer_relevance: float
    context_precision: float
    flagged_claims: list[str] = field(default_factory=list)
    broken_citations: list[str] = field(default_factory=list)
    citations_valid: bool = True