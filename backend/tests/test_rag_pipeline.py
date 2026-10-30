"""
Tests for the RAG pipeline components.

All external services (Gemini API, Qdrant) are mocked so these tests
run without real credentials or infrastructure.

Tests requiring real API credentials are clearly marked and skipped
when the environment variable GEMINI_API_KEY is not set.
"""
import pytest
import os
from unittest.mock import MagicMock, patch, PropertyMock


# ── RAGRetriever ──────────────────────────────────────────────────────────────

class TestRAGRetriever:
    """Tests for the retrieval component."""

    def _make_retriever(self, mock_chunks=None):
        from app.rag.retrieve import RAGRetriever

        embedder = MagicMock()
        embedder.embed_single.return_value = [0.1] * 768

        store = MagicMock()
        store.search.return_value = mock_chunks or [
            {
                "id": "chunk_1",
                "content": "Python is a programming language.",
                "metadata": {
                    "title": "Python Foundations",
                    "module": "Module 1",
                    "heading": "# Introduction",
                    "file_path": "module-01.md",
                },
                "score": 0.92,
            }
        ]

        return RAGRetriever(embedder=embedder, store=store, score_threshold=0.7)

    def test_retrieve_chunks_calls_embedder(self):
        retriever = self._make_retriever()
        retriever.retrieve_chunks("What is Python?")
        retriever.embedder.embed_single.assert_called_once_with("What is Python?")

    def test_retrieve_chunks_calls_store_search(self):
        retriever = self._make_retriever()
        retriever.retrieve_chunks("What is Python?")
        retriever.store.search.assert_called_once()

    def test_retrieve_chunks_returns_list(self):
        retriever = self._make_retriever()
        result = retriever.retrieve_chunks("What is Python?")
        assert isinstance(result, list)

    def test_retrieve_chunks_returns_expected_content(self):
        retriever = self._make_retriever()
        chunks = retriever.retrieve_chunks("What is Python?")
        assert chunks[0]["content"] == "Python is a programming language."

    def test_filter_by_evidence_removes_low_score_chunks(self):
        retriever = self._make_retriever()
        chunks = [
            {"id": "a", "content": "high", "score": 0.9},
            {"id": "b", "content": "low",  "score": 0.3},
            {"id": "c", "content": "mid",  "score": 0.75},
        ]
        filtered = retriever.filter_by_evidence(chunks, min_score=0.7)
        assert len(filtered) == 2
        assert all(c["score"] >= 0.7 for c in filtered)

    def test_check_sufficient_evidence_true_when_chunks_above_threshold(self):
        retriever = self._make_retriever()
        chunks = [{"id": "a", "content": "x", "score": 0.9}]
        assert retriever.check_sufficient_evidence(chunks, min_score=0.7) is True

    def test_check_sufficient_evidence_false_when_no_chunks(self):
        retriever = self._make_retriever()
        assert retriever.check_sufficient_evidence([], min_score=0.7) is False

    def test_check_sufficient_evidence_false_when_all_below_threshold(self):
        retriever = self._make_retriever()
        chunks = [{"id": "a", "content": "x", "score": 0.3}]
        assert retriever.check_sufficient_evidence(chunks, min_score=0.7) is False

    def test_format_citations_returns_list(self):
        retriever = self._make_retriever()
        chunks = [
            {
                "id": "chunk_1",
                "content": "Python is great.",
                "metadata": {
                    "title": "Python Foundations",
                    "module": "Module 1",
                    "heading": "# Intro",
                    "file_path": "module-01.md",
                },
                "score": 0.92,
            }
        ]
        citations = retriever.format_citations(chunks)
        assert isinstance(citations, list)
        assert len(citations) == 1

    def test_format_citations_has_title_field(self):
        retriever = self._make_retriever()
        chunks = [
            {
                "id": "c1",
                "content": "test",
                "metadata": {"title": "Python Foundations", "module": "M1",
                             "heading": "", "file_path": ""},
                "score": 0.9,
            }
        ]
        citations = retriever.format_citations(chunks)
        assert citations[0]["title"] == "Python Foundations"


# ── AnswerGenerator ───────────────────────────────────────────────────────────

class TestAnswerGenerator:
    """Tests for the answer generation component."""

    def _make_generator(self, answer_text="Python is a language."):
        from app.rag.generator import AnswerGenerator

        generator = AnswerGenerator(generation_model="gemini-2.5-flash", evidence_threshold=0.7)

        # Mock the Gemini client
        mock_response = MagicMock()
        mock_response.text = answer_text

        mock_genai = MagicMock()
        mock_genai.models.generate_content.return_value = mock_response

        with patch("app.rag.generator.genai", mock_genai):
            generator._mock_genai = mock_genai  # store for test inspection
        return generator, mock_genai

    def test_generate_answer_returns_dict(self):
        generator, mock_genai = self._make_generator()
        chunks = [{"id": "c1", "content": "Python is great.", "score": 0.9, "metadata": {}}]
        with patch("app.rag.generator.genai", mock_genai):
            result = generator.generate_answer("What is Python?", chunks)
        assert isinstance(result, dict)

    def test_generate_answer_has_answer_field(self):
        generator, mock_genai = self._make_generator()
        chunks = [{"id": "c1", "content": "Python is great.", "score": 0.9, "metadata": {}}]
        with patch("app.rag.generator.genai", mock_genai):
            result = generator.generate_answer("What is Python?", chunks)
        assert "answer" in result

    def test_generate_answer_returns_refusal_for_empty_chunks(self):
        generator, mock_genai = self._make_generator()
        with patch("app.rag.generator.genai", mock_genai):
            result = generator.generate_answer("What is Python?", chunks=[])
        assert result["answer"] == (
            "I can only answer questions using the content available in this AI/ML book. "
            "I could not find enough relevant information in the book to answer your question."
        )
        assert result["evidence_threshold_met"] is False

    def test_generate_answer_has_citations_field(self):
        generator, mock_genai = self._make_generator()
        chunks = [{"id": "c1", "content": "Python is great.", "score": 0.9, "metadata": {}}]
        with patch("app.rag.generator.genai", mock_genai):
            result = generator.generate_answer("What is Python?", chunks)
        assert "citations" in result

    def test_generate_answer_has_grounded_field(self):
        generator, mock_genai = self._make_generator()
        chunks = [{"id": "c1", "content": "Python is great.", "score": 0.9, "metadata": {}}]
        with patch("app.rag.generator.genai", mock_genai):
            result = generator.generate_answer("What is Python?", chunks)
        assert "grounded" in result


# ── RAGPipeline ────────────────────────────────────────────────────────────────

class TestRAGPipeline:
    """Integration-level tests for the full RAG pipeline (all external deps mocked)."""

    def _make_pipeline(self):
        from app.rag.pipeline import RAGPipeline

        # Patch all external calls at the class level
        with patch("app.rag.pipeline.GeminiEmbedder") as MockEmbedder, \
             patch("app.rag.pipeline.QdrantStore") as MockStore, \
             patch("app.rag.pipeline.RAGRetriever") as MockRetriever, \
             patch("app.rag.pipeline.AnswerGenerator") as MockGenerator, \
             patch("app.rag.pipeline.BookContentIndexer") as MockIndexer:

            mock_retriever_inst = MagicMock()
            mock_retriever_inst.retrieve_chunks.return_value = [
                {
                    "id": "c1",
                    "content": "Python is a programming language.",
                    "metadata": {"title": "Python Foundations", "module": "Module 1",
                                 "heading": "", "file_path": "module-01.md"},
                    "score": 0.95,
                }
            ]
            mock_retriever_inst.check_sufficient_evidence.return_value = True
            mock_retriever_inst.score_threshold = 0.7
            mock_retriever_inst.format_citations.return_value = [
                {"title": "Python Foundations", "module": "Module 1"}
            ]
            MockRetriever.return_value = mock_retriever_inst

            mock_generator_inst = MagicMock()
            mock_generator_inst.generate_answer.return_value = {
                "answer": "Python is a programming language.",
                "citations": [],
                "evidence_threshold_met": True,
                "grounded": True,
            }
            MockGenerator.return_value = mock_generator_inst

            mock_indexer_inst = MagicMock()
            mock_indexer_inst.ingest_all.return_value = {"files": 3, "chunks_indexed": 45, "errors": 0}
            MockIndexer.return_value = mock_indexer_inst

            pipeline = RAGPipeline()

        # Re-inject mocks after creation for test assertions
        pipeline.retriever = mock_retriever_inst
        pipeline.generator = mock_generator_inst
        pipeline.indexer = mock_indexer_inst
        return pipeline

    def test_answer_question_returns_dict(self):
        pipeline = self._make_pipeline()
        result = pipeline.answer_question("What is Python?")
        assert isinstance(result, dict)

    def test_answer_question_has_answer_field(self):
        pipeline = self._make_pipeline()
        result = pipeline.answer_question("What is Python?")
        assert "answer" in result

    def test_answer_question_has_citations_field(self):
        pipeline = self._make_pipeline()
        result = pipeline.answer_question("What is Python?")
        assert "citations" in result

    def test_answer_question_insufficient_evidence_returns_refusal(self):
        pipeline = self._make_pipeline()
        pipeline.retriever.check_sufficient_evidence.return_value = False
        result = pipeline.answer_question("What is quantum computing?")
        assert result["evidence_threshold_met"] is False
        assert result["classification"] == "out_of_scope"
        assert result["answer"] == (
            "I can only answer questions using the content available in this AI/ML book. "
            "I could not find enough relevant information in the book to answer your question."
        )

    def test_ingest_content_calls_indexer(self):
        pipeline = self._make_pipeline()
        pipeline.ingest_content(force_rebuild=False)
        pipeline.indexer.ingest_all.assert_called_once_with(force_rebuild=False)

    def test_ingest_content_returns_stats_dict(self):
        pipeline = self._make_pipeline()
        stats = pipeline.ingest_content()
        assert isinstance(stats, dict)
        assert "chunks_indexed" in stats

    def test_ingest_content_force_rebuild(self):
        pipeline = self._make_pipeline()
        pipeline.ingest_content(force_rebuild=True)
        pipeline.indexer.ingest_all.assert_called_once_with(force_rebuild=True)


# ── RAG Evaluation Dataset ─────────────────────────────────────────────────────

"""
Evaluation dataset for manual / CI evaluation of the RAG pipeline.

Questions marked 'in_scope' should be answerable from the book.
Questions marked 'out_of_scope' should trigger the refusal message.

Usage:
    from tests.test_rag_pipeline import EVALUATION_DATASET
    for item in EVALUATION_DATASET:
        result = pipeline.answer_question(item["question"])
        # Check expected_in_book vs evidence_threshold_met
"""

EVALUATION_DATASET = [
    # ── In-scope questions (should be answered) ──────────────────────────
    {
        "question": "What is Python?",
        "expected_in_book": True,
        "expected_topic": "Module 1",
        "notes": "Covered in Module 1",
    },
    {
        "question": "What is NumPy?",
        "expected_in_book": True,
        "expected_topic": "Module 2",
        "notes": "Covered in Module 2 - NumPy section",
    },
    {
        "question": "How does gradient descent work?",
        "expected_in_book": True,
        "expected_topic": "Module 3",
        "notes": "Covered in Module 3 - Calculus section",
    },
    {
        "question": "What is a Random Forest?",
        "expected_in_book": True,
        "expected_topic": "Module 4",
        "notes": "Covered in Module 4 - Classification",
    },
    {
        "question": "What is a convolutional neural network?",
        "expected_in_book": True,
        "expected_topic": "Module 5",
        "notes": "Covered in Module 5 - CNNs",
    },
    {
        "question": "What is MLflow used for?",
        "expected_in_book": True,
        "expected_topic": "Module 6",
        "notes": "Covered in Module 6 - MLOps",
    },
    {
        "question": "What is Apache Spark?",
        "expected_in_book": True,
        "expected_topic": "Module 7",
        "notes": "Covered in Module 7 - Big Data",
    },
    {
        "question": "How does RAG work?",
        "expected_in_book": True,
        "expected_topic": "Module 8",
        "notes": "Covered in Module 8 - RAG section",
    },
    {
        "question": "What is the ReAct framework for AI agents?",
        "expected_in_book": True,
        "expected_topic": "Module 9",
        "notes": "Covered in Module 9 - ReAct",
    },
    # ── Out-of-scope questions (should be refused) ───────────────────────
    {
        "question": "What is the capital of France?",
        "expected_in_book": False,
        "notes": "General knowledge, not covered in book",
    },
    {
        "question": "How do I bake a chocolate cake?",
        "expected_in_book": False,
        "notes": "Completely unrelated topic",
    },
    {
        "question": "Who won the 2024 FIFA World Cup?",
        "expected_in_book": False,
        "notes": "Sports news, not in book",
    },
    {
        "question": "Write me a poem about the ocean.",
        "expected_in_book": False,
        "notes": "Creative writing, not in book",
    },
]
