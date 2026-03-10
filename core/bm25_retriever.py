"""BM25 retriever for keyword-based search in legal documents."""
import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from core.models import DocumentChunk, RetrievedChunk

# Configuration
CACHE_DIR = Path(".cache")


class BM25Retriever:
    """BM25-based keyword retrieval for legal documents."""
    
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[DocumentChunk] = []
        self.cache_path = CACHE_DIR / "bm25_index.pkl"
        CACHE_DIR.mkdir(exist_ok=True)
        self._load_index()
    
    def add(self, chunks: list[DocumentChunk]) -> None:
        """Add new chunks to the BM25 index."""
        self.chunks.extend(chunks)
        self._rebuild_index()
        self._save_index()
    
    def search(
        self, 
        query: str, 
        top_k: int = 10, 
        filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Search using BM25 keyword matching."""
        if not self.bm25 or not self.chunks:
            return []
        
        # Tokenize query (simple whitespace split for now)
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Create scored chunks
        scored_chunks = [
            (chunk, score, idx) 
            for idx, (chunk, score) in enumerate(zip(self.chunks, scores))
            if score > 0
        ]
        
        # Apply filters if provided
        if filters:
            scored_chunks = [
                (chunk, score, idx)
                for chunk, score, idx in scored_chunks
                if self._matches_filters(chunk, filters)
            ]
        
        # Sort by score and take top_k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk, score, _ in scored_chunks[:top_k]:
            results.append(RetrievedChunk(
                chunk=chunk,
                score=float(score),
                retrieval_method="bm25",
            ))
        
        return results
    
    def _matches_filters(self, chunk: DocumentChunk, filters: dict[str, Any]) -> bool:
        """Check if chunk matches the provided filters."""
        for key, value in filters.items():
            if chunk.metadata.get(key) != value:
                return False
        return True
    
    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index from current chunks."""
        if not self.chunks:
            self.bm25 = None
            return
        
        # Tokenize all documents (simple whitespace split)
        tokenized_docs = [
            chunk.content.lower().split()
            for chunk in self.chunks
        ]
        
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def _save_index(self) -> None:
        """Save the BM25 index and chunks to cache."""
        try:
            cache_data = {
                "bm25": self.bm25,
                "chunks": self.chunks,
            }
            with open(self.cache_path, "wb") as f:
                pickle.dump(cache_data, f)
        except Exception:
            # If save fails, continue without caching
            pass
    
    def _load_index(self) -> None:
        """Load the BM25 index and chunks from cache."""
        try:
            if self.cache_path.exists():
                with open(self.cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                self.bm25 = cache_data.get("bm25")
                self.chunks = cache_data.get("chunks", [])
        except Exception:
            # If load fails, start fresh
            self.bm25 = None
            self.chunks = []
    
    def clear(self) -> None:
        """Clear the BM25 index and all chunks."""
        self.bm25 = None
        self.chunks = []
        if self.cache_path.exists():
            self.cache_path.unlink()