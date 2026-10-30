"""Gemini embedding integration.

Wraps the Google Gemini API for generating embeddings for documents and queries.
Model names verified against official Gemini documentation.
"""
import google.genai as genai
from google.genai import types
from typing import List, Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)


class GeminiEmbedder:
    """Wrapper for Gemini embedding API."""

    def __init__(
        self,
        embedding_model: str = "models/gemini-embedding-001",
        generation_model: str = "gemini-3.5-flash-lite",
        api_key: Optional[str] = None,
        output_dimension: int = 768,
    ):
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.api_key = api_key
        self.output_dimension = output_dimension
        # Gemini's free embedding tier is rate-limited by tokens/minute. Keep
        # requests spaced out across files as well as within a single batch.
        self._minimum_request_interval = 8.0
        self._last_request_at = 0.0

    def _initialize_client(self):
        """Initialize the Gemini client."""
        genai.configure(api_key=self.api_key)

    def _get_client(self):
        """Get Gemini client."""
        if self.api_key:
            return genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=30000),
            )
        return genai

    def embed_text(self, texts: List[str], throttle: bool = True) -> List[List[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        try:
            client = self._get_client()
            if throttle:
                elapsed = time.monotonic() - self._last_request_at
                if elapsed < self._minimum_request_interval:
                    time.sleep(self._minimum_request_interval - elapsed)
            # Use the new genai embeddings API
            result = client.models.embed_content(
                model=self.embedding_model,
                contents=texts,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.output_dimension,
                ),
            )
            self._last_request_at = time.monotonic()
            embeddings = result.embeddings
            # The result contains an embedding for each content in texts
            return [embedding.values for embedding in embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def embed_single(self, text: str) -> List[float]:
        """Generate an embedding for a single text."""
        # Query embeddings are tiny and should not wait for the ingestion
        # rate limiter; only bulk document ingestion needs throttling.
        return self.embed_text([text], throttle=False)[0]

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

    def embed_chunks_with_metadata(
        self, chunks: List[Dict[str, Any]], batch_size: int = 64
    ) -> List[Dict[str, Any]]:
        """Embed chunks in bounded batches using the same model and dimensions."""
        prepared: List[Dict[str, Any]] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = []
            for chunk in batch:
                metadata = chunk.get("metadata", {})
                parts = []
                if metadata.get("module"):
                    parts.append(f"Module: {metadata['module']}")
                heading = metadata.get("heading", metadata.get("section", ""))
                if heading:
                    parts.append(f"Section: {heading}")
                parts.append(chunk["content"])
                texts.append(" | ".join(parts))

            vectors = self.embed_text(texts)
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding API returned an unexpected number of vectors")
            for chunk, vector in zip(batch, vectors):
                if len(vector) != self.output_dimension:
                    raise ValueError(
                        f"Embedding dimension {len(vector)} does not match configured "
                        f"dimension {self.output_dimension}"
                    )
                item = chunk.copy()
                item["embedding"] = vector
                prepared.append(item)
        return prepared
