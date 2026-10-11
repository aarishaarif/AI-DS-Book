"""Main RAG pipeline tying together chunking, embedding, retrieval, and generation."""

from typing import List, Dict, Any, Optional
from .embeddings import GeminiEmbedder
from .store import QdrantStore
from .retrieve import RAGRetriever
from .generator import AnswerGenerator
from .ingest import BookContentIndexer


class RAGPipeline:
    """Complete Retrieval-Augmented Generation pipeline."""

    def __init__(
        self,
        *,
        embedding_model: str = "models/gemini-embedding-001",
        generation_model: str = "gemini-2.0-flash-exp",
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: str = "",
        book_dir: str = "../frontend/docs",
        evidence_threshold: float = 0.7,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        generation_temperature: float = 0.7,
        generation_max_tokens: int = 1024,
    ):
        # Initialize components
        self.embedder = GeminiEmbedder(
            embedding_model=embedding_model,
            generation_model=generation_model,
        )
        self.store = QdrantStore(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        self.retriever = RAGRetriever(
            embedder=self.embedder,
            store=self.store,
            limit=5,
            score_threshold=evidence_threshold,
        )
        self.generator = AnswerGenerator(
            generation_model=generation_model,
            evidence_threshold=evidence_threshold,
        )
        self.indexer = BookContentIndexer(
            embedder=self.embedder,
            store=self.store,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            book_dir=book_dir,
        )

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a user question using the full RAG pipeline.

        Args:
            question: User's question about the book content

        Returns:
            Dict with answer, citations, and evidence status
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve_chunks(question)

        # Check if sufficient evidence exists
        has_sufficient = self.retriever.check_sufficient_evidence(
            retrieved_chunks, min_score=self.retriever.score_threshold
        )

        if not has_sufficient:
            return {
                "answer": (
                    "I couldn't find sufficient information about this topic in the AI & Data Science Book. "
                    "Please try asking about a topic covered in the book."
                ),
                "citations": [],
                "evidence_threshold_met": False,
                "grounded": False,
            }

        # Generate answer from retrieved chunks
        result = self.generator.generate_answer(
            question=question,
            chunks=retrieved_chunks,
        )

        # Format citations
        citations = self.retriever.format_citations(retrieved_chunks)
        result["citations"] = citations

        return result

    def ingest_content(self, force_rebuild: bool = False) -> Dict[str, int]:
        """Index all book content into the vector store.

        Args:
            force_rebuild: If True, clear the collection before indexing

        Returns:
            Dict with indexing statistics
        """
        return self.indexer.ingest_all(force_rebuild=force_rebuild)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        total = self.store.get_total_count()
        return {"total_chunks": total}


# Convenience function for FastAPI route
def create_rag_pipeline(**kwargs) -> RAGPipeline:
    """Create a RAGPipeline instance with given configuration."""
    return RAGPipeline(**kwargs)