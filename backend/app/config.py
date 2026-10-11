"""
Configuration module for the AI & Data Science Learning Book backend.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # Gemini API configuration
    gemini_embedding_model: str = (
        "models/gemini-embedding-001"  # Official Gemini embedding model
    )
    gemini_generation_model: str = (
        "gemini-2.0-flash-exp"  # Official Gemini generation model
    )

    gemini_api_key: str = ""
    gemini_embedding_api_key: str = ""

    # Qdrant configuration
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # Application configuration
    host: str = "0.0.0.0"
    port: int = 8000
    chunk_size: int = 800
    chunk_overlap: int = 100
    evidence_threshold: float = 0.7

    # File paths
    book_dir: str = "../frontend/docs"  # Directory containing book content


# Global settings instance
_settings = Settings()


def get_settings() -> Settings:
    """Get global settings instance."""
    return _settings


def get_gemini_embedder():
    """Get configured Gemini embedder client."""
    pass


def get_qdrant_client():
    """Get configured Qdrant client."""
    pass


def get_chunker():
    """Get configured text chunker."""
    pass