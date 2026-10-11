"""Document retrieval for RAG pipeline.

Selects relevant document chunks for answering user questions.
"""
from typing import List, Dict, Any, Optional
from .embeddings import GeminiEmbedder
from .store import QdrantStore
from .chunker import chunk_by_tokens


class RAGRetriever:
    """Handles retrieval of relevant document chunks for user queries."""

    def __init__(
        self,
        embedder: GeminiEmbedder,
        store: QdrantStore,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        limit: int = 5,
        score_threshold: float = 0.7,
    ):
        self.embedder = embedder
        self.store = store
        self.limit = limit
        self.score_threshold = score_threshold

    def retrieve_chunks(
        self,
        question: str,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a question.

        Args:
            question: User's question to embed and search for
            filter_metadata: Optional metadata to filter search results

        Returns:
            List of retrieved chunks with scores and metadata
        """
        # Generate embedding for the question
        query_embedding = self.embedder.embed_single(question)

        # Search for similar chunks in the vector store
        retrieved_chunks = self.store.search(
            query_embedding=query_embedding,
            limit=self.limit,
            score_threshold=self.score_threshold,
            filter_metadata=filter_metadata,
        )

        return retrieved_chunks

    def format_citations(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format chunk metadata as clickable citations.

        Args:
            chunks: List of retrieved chunks

        Returns:
            Formatted citation list
        """
        citations = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            citation = {
                "title": metadata.get("title", "Untitled"),
                "module": metadata.get("module", ""),
                "section": metadata.get("heading", metadata.get("section", "")),
                "lesson": metadata.get("lesson", ""),
                "file_path": metadata.get("file_path", ""),
                "chunk_id": chunk.get("id", ""),
                "relevance_score": chunk.get("score", 0.0),
                "text_excerpt": chunk["content"][:200] + "..."
                if len(chunk["content"]) > 200
                else chunk["content"],
            }
            citations.append(citation)
        return citations

    def filter_by_evidence(
        self, chunks: List[Dict[str, Any]], min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Filter retrieved chunks by relevance score.

        Args:
            chunks: List of retrieved chunks
            min_score: Minimum relevance score threshold

        Returns:
            Filtered list of chunks meeting the threshold
        """
        return [chunk for chunk in chunks if chunk.get("score", 0.0) >= min_score]

    def check_sufficient_evidence(
        self, chunks: List[Dict[str, Any]], min_score: float = 0.7, min_chunks: int = 1
    ) -> bool:
        """Check if retrieved chunks provide sufficient evidence for answer generation.

        Args:
            chunks: List of retrieved chunks
            min_score: Minimum relevance score threshold
            min_chunks: Minimum number of chunks required

        Returns:
            True if sufficient evidence, False otherwise
        """
        filtered_chunks = self.filter_by_evidence(chunks, min_score)
        return len(filtered_chunks) >= min_chunks