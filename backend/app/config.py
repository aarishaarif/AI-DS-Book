"""
Configuration module for the AI & Data Science Learning Book backend.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        case_sensitive=False,
        extra="ignore",
    )

    # Gemini API configuration
    gemini_embedding_model: str = (
        "models/gemini-embedding-001"  # Official Gemini embedding model
    )
    gemini_generation_model: str = (
        "gemini-3.5-flash-lite"  # Low-latency Gemini generation model
    )

    gemini_api_key: str = ""
    gemini_embedding_api_key: str = ""

    # Qdrant configuration. Set these in backend/.env or the process
    # environment. MCP server variables are not inherited automatically.
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "ai_ds_book"

    # Gemini Embedding 001 supports 128–3072 dimensions. 768 is a documented
    # recommended size and is explicitly requested for both indexing and query
    # embeddings so the collection size is never guessed by the store layer.
    embedding_dimension: int = 768

    # Application configuration
    host: str = "0.0.0.0"
    port: int = 8000
    chunk_size: int = 800
    chunk_overlap: int = 100
    evidence_threshold: float = 0.7
    retrieval_limit: int = 6

    # File paths
    book_dir: str = "../frontend/docs"  # Directory containing book content


# Global settings instance
_settings = Settings()


def get_settings() -> Settings:
    """Get global settings instance."""
    return _settings


def get_gemini_embedder():
    """Get configured Gemini embedder client."""
    from .rag.embeddings import GeminiEmbedder

    settings = get_settings()
    return GeminiEmbedder(
        embedding_model=settings.gemini_embedding_model,
        generation_model=settings.gemini_generation_model,
        api_key=settings.gemini_api_key or settings.gemini_embedding_api_key,
        output_dimension=settings.embedding_dimension,
    )


def get_qdrant_client():
    """Get configured Qdrant client."""
    from qdrant_client import QdrantClient

    settings = get_settings()
    if not settings.qdrant_url.strip():
        raise RuntimeError(
            "QDRANT_URL is not configured. Set the Qdrant Cloud URL in backend/.env "
            "or the backend process environment."
        )
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


def get_chunker():
    """Get configured text chunker."""
    from .rag.chunker import chunk_by_tokens

    settings = get_settings()
    return lambda content, metadata=None: chunk_by_tokens(
        content,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        metadata=metadata,
    )
