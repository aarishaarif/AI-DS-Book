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
        collection_name: str = "ai_ds_book",
        vector_size: int = 768,
    ):
        if not url or not url.strip():
            raise ValueError(
                "QDRANT_URL is not configured. Set the Qdrant Cloud URL in backend/.env "
                "or the backend process environment."
            )
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection_exists()

    @staticmethod
    def _qdrant_point_id(point_id: Any) -> str:
        """Convert an application chunk ID into a valid deterministic Qdrant ID.

        Qdrant point IDs must be unsigned integers or UUIDs. Chunk IDs are
        intentionally human-readable (for example ``chunk_ab12...``), so keep
        those IDs in the payload and use a stable UUID for the actual point.
        """
        candidate = str(point_id or uuid.uuid4())
        try:
            return str(uuid.UUID(candidate))
        except (ValueError, AttributeError):
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ai-ds-learning-book:{candidate}"))

    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        try:
            exists = self.client.collection_exists(collection_name=self.collection_name)
        except Exception:
            # Connection and authentication failures must remain visible to the
            # caller; they must not be mistaken for a missing collection.
            raise
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.vector_size,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            return

        info = self.client.get_collection(collection_name=self.collection_name)
        vectors = info.config.params.vectors
        # The application uses unnamed, dense vectors. Refuse to write into a
        # collection whose existing schema would mix incompatible embeddings.
        if not isinstance(vectors, qdrant_models.VectorParams):
            raise ValueError(
                f"Collection {self.collection_name!r} does not use a single unnamed dense vector"
            )
        if vectors.size != self.vector_size or vectors.distance != qdrant_models.Distance.COSINE:
            raise ValueError(
                f"Collection {self.collection_name!r} is configured for size={vectors.size}, "
                f"distance={vectors.distance}; expected size={self.vector_size}, distance=Cosine. "
                "Use a separate collection and reindex rather than mixing vector configurations."
            )

    def index_chunk(self, chunk: Dict[str, Any]) -> str:
        """Index a single chunk with its embedding.

        Args:
            chunk: Chunk dict with 'id', 'content', 'metadata', and 'embedding'

        Returns:
            Success status
        """
        original_id = chunk.get("id", str(uuid.uuid4()))
        point_id = self._qdrant_point_id(original_id)
        point = qdrant_models.PointStruct(
            id=point_id,
            vector=chunk["embedding"],
            payload={
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "original_id": original_id,
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
            original_id = chunk.get("id", str(uuid.uuid4()))
            point_id = self._qdrant_point_id(original_id)
            point = qdrant_models.PointStruct(
                id=point_id,
                vector=chunk["embedding"],
                payload={
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "original_id": original_id,
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
        """Remove points tagged as this book dataset while preserving the collection."""
        self.delete_by_document_ids(self.list_document_ids())

    def delete_by_document_ids(self, document_ids: List[str]) -> int:
        """Delete stale points belonging to known book documents only."""
        if not document_ids:
            return 0
        selector = qdrant_models.FilterSelector(
            filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.dataset",
                        match=qdrant_models.MatchValue(value="ai_ds_book"),
                    ),
                    qdrant_models.FieldCondition(
                        key="metadata.source_document_id",
                        match=qdrant_models.MatchAny(any=document_ids),
                    )
                ]
            )
        )
        self.client.delete(collection_name=self.collection_name, points_selector=selector)
        return len(document_ids)

    def replace_document(self, document_id: str, chunks: List[Dict[str, Any]]) -> int:
        """Upsert a complete document version, then remove its stale chunk IDs."""
        old_points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.dataset",
                        match=qdrant_models.MatchValue(value="ai_ds_book"),
                    ),
                    qdrant_models.FieldCondition(
                    key="metadata.source_document_id",
                    match=qdrant_models.MatchValue(value=document_id),
                    ),
                ]
            ),
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        if chunks:
            indexed, new_ids = self.index_chunks(chunks)
        else:
            indexed, new_ids = 0, []
        stale_ids = [point.id for point in old_points if point.id not in set(new_ids)]
        if stale_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.PointIdsList(points=stale_ids),
            )
        return indexed

    def list_document_ids(self) -> List[str]:
        """Return document ids currently owned by this book dataset."""
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qdrant_models.Filter(
                must=[qdrant_models.FieldCondition(
                    key="metadata.dataset",
                    match=qdrant_models.MatchValue(value="ai_ds_book"),
                )]
            ),
            limit=10000,
            with_payload=["metadata"],
            with_vectors=False,
        )
        return sorted({
            point.payload.get("metadata", {}).get("source_document_id")
            for point in points
            if point.payload and point.payload.get("metadata", {}).get("source_document_id")
        })

    def collection_info(self):
        return self.client.get_collection(collection_name=self.collection_name)

    def health_check(self) -> bool:
        self.client.get_collections()
        return True

    def get_total_count(self) -> int:
        """Get total number of chunks in the collection."""
        info = self.client.get_collection(collection_name=self.collection_name)
        return info.points_count
