import os
from dotenv import load_dotenv
import cohere

from core.bm25_retriever import BM25Retriever
from core.models import DocumentChunk, RetrievedChunk
from core.vector_store import QdrantVectorStore

# Load environment variables
load_dotenv()

# Environment variables
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
RERANKER_MODEL = "rerank-v4.0"
TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 5


class HybridRetriever:
    """
    Merges vector + BM25 results via Reciprocal Rank Fusion,
    then reranks top candidates with a cross-encoder.
    """

    def __init__(self, vector_store: QdrantVectorStore, bm25: BM25Retriever):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.cohere_client = cohere.Client(api_key=COHERE_API_KEY)
        self.reranker_model = RERANKER_MODEL
        self.top_k_retrieval = TOP_K_RETRIEVAL
        self.top_k_rerank = TOP_K_RERANK

    def retrieve(
        self, query: str, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        vector_results = self.vector_store.search(query, self.top_k_retrieval, filters)
        bm25_results = self.bm25.search(query, self.top_k_retrieval)

        fused = self._reciprocal_rank_fusion(vector_results, bm25_results)
        return self._rerank(query, fused[: self.top_k_retrieval])

    def _reciprocal_rank_fusion(
        self,
        list_a: list[RetrievedChunk],
        list_b: list[RetrievedChunk],
        k: int = 60,
    ) -> list[RetrievedChunk]:
        scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank, item in enumerate(list_a):
            cid = item.chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
            chunk_map[cid] = item

        for rank, item in enumerate(list_b):
            cid = item.chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
            chunk_map[cid] = item

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            RetrievedChunk(
                chunk=chunk_map[cid].chunk,
                score=score,
                retrieval_method="hybrid",
            )
            for cid, score in ranked
        ]

    def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []

        try:
            response = self.cohere_client.rerank(
                query=query,
                documents=[c.chunk.content for c in candidates],
                model=self.reranker_model,
                top_n=self.top_k_rerank,
            )
            return [
                RetrievedChunk(
                    chunk=candidates[r.index].chunk,
                    score=r.relevance_score,
                    retrieval_method="reranked",
                )
                for r in response.results
            ]
        except Exception:
            # Fallback to fusion scores if Cohere is unavailable
            return candidates[: self.top_k_rerank]
