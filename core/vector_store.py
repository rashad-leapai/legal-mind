"""Qdrant vector store implementation for LegalMind."""
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from core.models import DocumentChunk, RetrievedChunk

# Load environment variables
load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "legal_documents"
EMBEDDING_MODEL = "text-embedding-3-large"


class QdrantVectorStore:
    """Vector store using Qdrant for legal document embeddings."""
    
    def __init__(self):
        # Initialize OpenAI client for embeddings
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize Qdrant Cloud client
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        
        self.collection_name = QDRANT_COLLECTION_NAME
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            # Collection doesn't exist, create it
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=3072,  # text-embedding-3-large dimension
                    distance=Distance.COSINE,
                ),
            )
    
    def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding using OpenAI."""
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding
    
    def add(self, chunks: list[DocumentChunk]) -> None:
        """Add document chunks to the vector store."""
        points = []
        
        for chunk in chunks:
            if chunk.embedding is None:
                chunk.embedding = self._get_embedding(chunk.content)
            
            point = PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                },
            )
            points.append(point)
        
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
    
    def search(
        self, 
        query: str, 
        top_k: int = 10, 
        filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Search for relevant chunks using vector similarity."""
        query_embedding = self._get_embedding(query)
        
        # Build Qdrant filter if provided
        qdrant_filter = None
        if filters:
            qdrant_filter = self._build_filter(filters)
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding, 
            limit=top_k,
            query_filter=qdrant_filter,
        )
        
        retrieved_chunks = []
        for point in results.points:
            chunk = DocumentChunk(
                chunk_id=point.payload["chunk_id"],
                doc_id=point.payload["doc_id"],
                content=point.payload["content"],
                metadata=point.payload["metadata"],
                embedding=None,  # Don't store embeddings in memory
            )
            
            retrieved_chunks.append(RetrievedChunk(
                chunk=chunk,
                score=point.score,
                retrieval_method="vector",
            ))
        
        return retrieved_chunks
    
    def _build_filter(self, filters: dict[str, Any]):
        """Build Qdrant filter from dict filters with support for date ranges and multiple values."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range
        
        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                # Multiple values (OR condition)
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchAny(any=value),
                    )
                )
            elif isinstance(value, dict) and "start" in value and "end" in value:
                # Date range filter (for future date-based filtering)
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        range=Range(
                            gte=value["start"],
                            lte=value["end"]
                        )
                    )
                )
            else:
                # Single value match
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value),
                    )
                )
        
        if conditions:
            return Filter(must=conditions)
        return None
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
        except Exception:
            pass