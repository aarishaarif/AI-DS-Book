"""Gemini embedding integration.

Wraps the Google Gemini API for generating embeddings for documents and queries.
Model names verified against official Gemini documentation.
"""
import google.genai as genai
from google.genai import types
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class GeminiEmbedder:
    """Wrapper for Gemini embedding API."""

    def __init__(
        self,
        embedding_model: str = "models/gemini-embedding-001",
        generation_model: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
    ):
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.api_key = api_key

    def _initialize_client(self):
        """Initialize the Gemini client."""
        genai.configure(api_key=self.api_key)
        self.client = genai

    def _get_client(self):
        """Get or create Gemini client."""
        if not genai._client:
            self._initialize_client()
        return genai.client

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        try:
            client = self._get_client()
            # Use the new genai embeddings API
            result = client.models.embed_content(
                model=self.embedding_model,
                contents=texts,
            )
            embeddings = result.embeddings
            # The result contains an embedding for each content in texts
            return [embedding.values for embedding in embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def embed_single(self, text: str) -> List[float]:
        """Generate an embedding for a single text."""
        return self.embed_text([text])[0]

    def embed_chunk_with_metadata(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embedding for a chunk including its metadata in the text.

        This helps the retriever consider document context when ranking results.
        """
        metadata = chunk.get("metadata", {})
        # Include heading and module context in the embed text for better retrieval
        embed_parts = []
        heading = metadata.get("heading", metadata.get("section", ""))
        module = metadata.get("module", "")

        if module:
            embed_parts.append(f"Module: {module}")
        if heading:
            embed_parts.append(f"Section: {heading}")

        embed_context = " | ".join(embed_parts)
        if embed_context:
            embed_context += " | "
        embed_context += chunk["content"]

        embedding = self.embed_single(embed_context)
        chunk_with_embedding = chunk.copy()
        chunk_with_embedding["embedding"] = embedding
        return chunk_with_embedding