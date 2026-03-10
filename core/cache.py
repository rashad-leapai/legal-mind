import hashlib
import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from core.models import RAGResponse

# Load environment variables
load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")
CACHE_SIMILARITY_THRESHOLD = 0.85  # Cosine similarity threshold for cache hits

logger = logging.getLogger(__name__)


class SemanticCache:
    """Redis-based semantic cache with embedding similarity search."""

    def __init__(self):
        self.redis_client = None
        self.openai_client = None
        self.available = False
        
        try:
            if REDIS_URL:
                import redis
                self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
                self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
                # Test connection
                self.redis_client.ping()
                self.available = True
                logger.info("Semantic cache enabled with Redis Cloud")
            else:
                logger.info("Semantic cache disabled (no Redis URL configured)")
        except Exception as e:
            logger.warning(f"Failed to initialize semantic cache: {e}")
            self.available = False

    def get(self, query: str) -> Optional[RAGResponse]:
        """Get cached response for semantically similar queries."""
        if not self.available:
            return None
        
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            
            # Search for similar cached queries
            cache_keys = self.redis_client.keys("legalmind:cache:*")
            
            for key in cache_keys:
                cached_data = self.redis_client.hgetall(key)
                if not cached_data:
                    continue
                
                # Compare embeddings
                cached_embedding = json.loads(cached_data.get("embedding", "[]"))
                if not cached_embedding:
                    continue
                
                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                
                if similarity >= CACHE_SIMILARITY_THRESHOLD:
                    logger.info(f"Cache hit for query similarity: {similarity:.3f}")
                    
                    # Reconstruct RAGResponse
                    response_data = json.loads(cached_data["response"])
                    response = RAGResponse(
                        answer=response_data["answer"],
                        source_ids=response_data["source_ids"],
                        retrieved_chunks=[],  # Don't cache heavy chunk data
                        cached=True
                    )
                    return response
            
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None

    def set(self, query: str, response: RAGResponse) -> None:
        """Cache response with query embedding for semantic search."""
        if not self.available:
            return
        
        try:
            # Generate cache key
            cache_key = f"legalmind:cache:{self._key(query)}"
            
            # Get query embedding
            query_embedding = self._get_embedding(query)
            
            # Prepare cache data
            cache_data = {
                "query": query,
                "embedding": json.dumps(query_embedding),
                "response": json.dumps({
                    "answer": response.answer,
                    "source_ids": response.source_ids,
                }),
                "timestamp": str(int(__import__("time").time()))
            }
            
            # Store in Redis with 1 hour TTL
            self.redis_client.hset(cache_key, mapping=cache_data)
            self.redis_client.expire(cache_key, 3600)  # 1 hour
            
            logger.debug(f"Cached response for query: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Cache set failed: {e}")

    def _key(self, query: str) -> str:
        """Generate deterministic cache key from query."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for semantic comparison."""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",  # Smaller model for cache efficiency
            input=text
        )
        return response.data[0].embedding
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        import math
        
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(x * x for x in b))
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        return dot_product / (magnitude_a * magnitude_b)
