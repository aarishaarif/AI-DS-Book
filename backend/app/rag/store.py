"""Qdrant vector store integration.

Handles storage and retrieval of document chunks with vector embeddings.
"""
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
import uuid


class QdrantStore:
    """Qdrant vector store for RAG pipeline."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str = "",
        collection_name: str = "ai_ds_learning_book",
    ):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        try:
            self.client.get_collection(collection_name=self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=768,  # Default for Gemini embeddings (verify if different)
                    distance=qdrant_models.Distance.COSINE,
                ),
            )

    def index_chunk(self, chunk: Dict[str, Any]) -> str:
        """Index a single chunk with its embedding.

        Args:
            chunk: Chunk dict with 'id', 'content', 'metadata', and 'embedding'

        Returns:
            Success status
        """
        point_id = chunk.get("id", str(uuid.uuid4()))
        point = qdrant_models.PointStruct(
            id=point_id,
            vector=chunk["embedding"],
            payload={
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "original_id": point_id,
            },
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        return point_id

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """Index multiple chunks.

        Args:
            chunks: List of chunk dicts

        Returns:
            Tuple of (number of successfully indexed chunks, list of successful IDs)
        """
        points = []
        for chunk in chunks:
            point_id = chunk.get("id", str(uuid.uuid4()))
            point = qdrant_models.PointStruct(
                id=point_id,
                vector=chunk["embedding"],
                payload={
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "original_id": point_id,
                },
            )
            points.append(point)

        result = self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        return (
            len(points),
            [p.id for p in points],
        )

    def search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        score_threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity.

        Args:
            query_embedding: Vector embedding of the query
            limit: Maximum number of results to return
            score_threshold: Minimum relevance score threshold
            filter_metadata: Optional filter to restrict search to specific metadata

        Returns:
            List of retrieved chunks with metadata and similarity scores
        """
        query_filter = None
        if filter_metadata:
            query_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.module",
                        match=qdrant_models.MatchValue(
                            value=filter_metadata.get("module")
                        ),
                    )
                ]
            )

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

        chunks = []
        for point in search_result.points:
            chunk = {
                "id": point.id,
                "content": point.payload["content"],
                "metadata": point.payload["metadata"],
                "score": point.score,
            }
            chunks.append(chunk)

        return chunks

    def clear_collection(self):
        """Clear all documents from the collection (for reindexing)."""
        self.client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection_exists()

    def get_total_count(self) -> int:
        """Get total number of chunks in the collection."""
        info = self.client.get_collection(collection_name=self.collection_name)
        return info.points_count