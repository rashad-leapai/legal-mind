from pathlib import Path

from core.bm25_retriever import BM25Retriever
from core.cache import SemanticCache
from core.generation import GenerationLayer
from core.ingestion import IngestionPipeline
from core.models import RAGResponse
from core.retriever import HybridRetriever
from core.vector_store import QdrantVectorStore


class LegalMindRAG:
    """
    Orchestrates the full RAG pipeline. Components are injected to allow
    swapping (e.g., replace ChromaVectorStore with Pinecone without touching this class).
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore | None = None,
        bm25: BM25Retriever | None = None,
        cache: SemanticCache | None = None,
        generation: GenerationLayer | None = None,
    ):
        self.ingestion = IngestionPipeline()
        self.vector_store = vector_store or QdrantVectorStore()
        self.bm25 = bm25 or BM25Retriever()
        self.retriever = HybridRetriever(self.vector_store, self.bm25)
        self.cache = cache or SemanticCache()
        self.generation = generation or GenerationLayer()

    def ingest_document(self, file_path: Path) -> int:
        chunks = self.ingestion.ingest(file_path)
        self.vector_store.add(chunks)
        self.bm25.add(chunks)
        return len(chunks)

    def query(self, question: str, filters: dict | None = None) -> RAGResponse:
        cached = self.cache.get(question)
        if cached:
            return cached

        chunks = self.retriever.retrieve(question, filters)
        response = self.generation.generate(question, chunks)

        self.cache.set(question, response)
        return response
