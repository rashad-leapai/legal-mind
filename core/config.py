"""Configuration settings for LegalMind RAG system."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    """Application settings."""
    # OpenAI API
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-large"
    
    # Vector Database
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    collection_name: str = "legal_documents"
    
    # Reranking
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    rerank_model: str = "rerank-v4.0"
    
    # Redis Cache
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_password: str = os.getenv("REDIS_PASSWORD", "")
    
    # Evaluation Thresholds
    faithfulness_threshold: float = 0.9
    relevance_threshold: float = 0.8
    precision_threshold: float = 0.85


_settings = None

def get_settings() -> Settings:
    """Get application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings