"""Main RAG pipeline tying together chunking, embedding, retrieval, and generation."""

from typing import List, Dict, Any, Optional
from .embeddings import GeminiEmbedder
from .store import QdrantStore
from .retrieve import RAGRetriever
from .generator import AnswerGenerator
from .ingest import BookContentIndexer
from .constants import REFUSAL_MESSAGE


class RAGPipeline:
    """Complete Retrieval-Augmented Generation pipeline."""

    def __init__(
        self,
        *,
        embedding_model: str = "models/gemini-embedding-001",
        generation_model: str = "gemini-3.5-flash-lite",
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: str = "",
        book_dir: str = "../frontend/docs",
        evidence_threshold: float = 0.7,
        embedding_dimension: int = 768,
        collection_name: str = "ai_ds_book",
        retrieval_limit: int = 6,
        api_key: str = "",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        generation_temperature: float = 0.7,
        generation_max_tokens: int = 1024,
    ):
        # Initialize components
        self.embedder = GeminiEmbedder(
            embedding_model=embedding_model,
            generation_model=generation_model,
            api_key=api_key,
            output_dimension=embedding_dimension,
        )
        self.store = QdrantStore(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
            vector_size=embedding_dimension,
        )
        self.retriever = RAGRetriever(
            embedder=self.embedder,
            store=self.store,
            limit=retrieval_limit,
            score_threshold=evidence_threshold,
        )
        self.generator = AnswerGenerator(
            generation_model=generation_model,
            api_key=api_key,
            evidence_threshold=evidence_threshold,
        )
        self.indexer = BookContentIndexer(
            embedder=self.embedder,
            store=self.store,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            book_dir=book_dir,
        )

    def answer_question(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Answer a user question using the full RAG pipeline.

        Args:
            question: User's question about the book content

        Returns:
            Dict with answer, citations, and evidence status
        """
        normalized = question.strip().lower()
        if normalized in {"hi", "hello", "hey", "salam", "assalamualaikum", "good morning", "good evening"}:
            return {
                "answer": "Hello! Ask me about a concept covered in the AI & Data Science Learning Book.",
                "citations": [], "evidence_threshold_met": True, "grounded": True,
            }

        # Retrieval is performed before scope classification. This supports
        # natural questions such as "What is a variable?" even when the exact
        # module name is not present in the wording.
        scope_terms = (
            "python", "numpy", "pandas", "visualization", "statistics", "probability",
            "linear algebra", "calculus", "machine learning", "regression", "classification",
            "clustering", "neural", "deep learning", "cnn", "rnn", "mlops", "mlflow",
            "spark", "big data", "cloud", "kafka", "llm", "language model", "transformer",
            "embedding", "prompt", "rag", "fine tuning", "agent", "react", "tool calling",
            "fastapi", "docker", "data science",
        )
        try:
            retrieved_chunks = self.retriever.retrieve_chunks(question)
        except Exception:
            return {
                "answer": "I'm temporarily unable to search the book. Please try again shortly.",
                "citations": [], "evidence_threshold_met": False, "grounded": False,
                "classification": "service_error",
            }

        # Check if sufficient evidence exists
        has_sufficient = self.retriever.check_sufficient_evidence(
            retrieved_chunks, min_score=self.retriever.score_threshold
        )

        if not has_sufficient:
            classification = "missing_content" if any(term in normalized for term in scope_terms) else "out_of_scope"
            return {
                "answer": REFUSAL_MESSAGE,
                "citations": [],
                "evidence_threshold_met": False,
                "grounded": False,
                "classification": classification,
            }

        # Generate answer from retrieved chunks
        result = self.generator.generate_answer(
            question=question,
            chunks=retrieved_chunks,
            history=history or [],
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
