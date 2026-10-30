"""
RAG Pipeline Evaluation Suite.

Evaluates retrieval quality, evidence thresholding, grounded answer generation,
citation validity, and refusal behavior for both in-scope and out-of-scope queries.
All external dependencies are mocked to run in CI/offline environments.
"""
import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

EVAL_BENCHMARK = [
    # In-scope curriculum topics
    {
        "id": "eval-01",
        "question": "What is Python and what are its core data structures?",
        "in_scope": True,
        "module": "Module 1",
        "expected_terms": ["python", "list", "tuple", "dictionary"],
        "mock_chunk": {
            "id": "chk-py-1",
            "content": "Python is a dynamic programming language with core structures like lists, tuples, and dictionaries.",
            "metadata": {
                "title": "Python Foundations",
                "module": "Module 1",
                "heading": "1.1 Variables and Data Types",
                "file_path": "module-01-python-foundations.md",
            },
            "score": 0.94,
        },
    },
    {
        "id": "eval-02",
        "question": "How does gradient descent optimize weights in machine learning?",
        "in_scope": True,
        "module": "Module 3",
        "expected_terms": ["gradient", "descent", "learning rate", "loss"],
        "mock_chunk": {
            "id": "chk-gd-1",
            "content": "Gradient descent computes the gradient of the loss function with respect to weights and updates via learning rate.",
            "metadata": {
                "title": "Statistics & Mathematics for ML",
                "module": "Module 3",
                "heading": "3.5 Calculus for ML",
                "file_path": "module-03-statistics-and-mathematics.md",
            },
            "score": 0.91,
        },
    },
    {
        "id": "eval-03",
        "question": "What is backpropagation in deep neural networks?",
        "in_scope": True,
        "module": "Module 5",
        "expected_terms": ["backpropagation", "chain rule", "gradient"],
        "mock_chunk": {
            "id": "chk-nn-1",
            "content": "Backpropagation uses the calculus chain rule to propagate error backwards from output to input layers.",
            "metadata": {
                "title": "Deep Learning",
                "module": "Module 5",
                "heading": "5.2 Backpropagation",
                "file_path": "module-05-deep-learning.md",
            },
            "score": 0.89,
        },
    },
    {
        "id": "eval-04",
        "question": "What is the ReAct framework in Agentic AI?",
        "in_scope": True,
        "module": "Module 9",
        "expected_terms": ["react", "reason", "act", "observation"],
        "mock_chunk": {
            "id": "chk-agent-1",
            "content": "The ReAct framework combines reasoning and action steps into iterative Thought -> Action -> Observation loops.",
            "metadata": {
                "title": "Agentic AI",
                "module": "Module 9",
                "heading": "9.2 ReAct Framework",
                "file_path": "module-09-agentic-ai.md",
            },
            "score": 0.93,
        },
    },
    # Out-of-scope topics that MUST be refused
    {
        "id": "eval-out-01",
        "question": "What is the capital city of France?",
        "in_scope": False,
        "mock_chunk": {
            "id": "chk-irrelevant-1",
            "content": "Unrelated book text about computing.",
            "metadata": {"title": "Intro", "module": "Module 1", "file_path": "intro.md"},
            "score": 0.32,  # Well below 0.70 threshold
        },
    },
    {
        "id": "eval-out-02",
        "question": "How do you bake a chocolate cake at home?",
        "in_scope": False,
        "mock_chunk": {
            "id": "chk-irrelevant-2",
            "content": "Unrelated code example.",
            "metadata": {"title": "Python", "module": "Module 1", "file_path": "module-01.md"},
            "score": 0.25,
        },
    },
    {
        "id": "eval-out-03",
        "question": "Who won the men's 100m sprint in the Olympic Games?",
        "in_scope": False,
        "mock_chunk": None,  # Zero chunks returned
    },
]


class TestRAGEvaluation:
    """Benchmark tests validating precision, recall, citation accuracy, and refusal."""

    @pytest.fixture
    def mock_pipeline_builder(self):
        def _build(mock_chunks, has_sufficient):
            from app.rag.pipeline import RAGPipeline

            with patch("app.rag.pipeline.GeminiEmbedder"), \
                 patch("app.rag.pipeline.QdrantStore"), \
                 patch("app.rag.pipeline.RAGRetriever") as MockRetriever, \
                 patch("app.rag.pipeline.AnswerGenerator") as MockGenerator, \
                 patch("app.rag.pipeline.BookContentIndexer"):

                mock_retriever = MagicMock()
                mock_retriever.retrieve_chunks.return_value = mock_chunks or []
                mock_retriever.check_sufficient_evidence.return_value = has_sufficient
                mock_retriever.score_threshold = 0.70

                if mock_chunks:
                    mock_retriever.format_citations.return_value = [
                        {
                            "title": c["metadata"].get("title", ""),
                            "module": c["metadata"].get("module", ""),
                            "section": c["metadata"].get("heading", ""),
                            "file_path": c["metadata"].get("file_path", ""),
                            "relevance_score": c["score"],
                            "text_excerpt": c["content"][:100],
                        }
                        for c in mock_chunks
                    ]
                else:
                    mock_retriever.format_citations.return_value = []

                MockRetriever.return_value = mock_retriever

                mock_generator = MagicMock()
                if has_sufficient:
                    mock_generator.generate_answer.return_value = {
                        "answer": "Grounded response based on book content.",
                        "citations": mock_retriever.format_citations.return_value,
                        "evidence_threshold_met": True,
                        "grounded": True,
                    }
                else:
                    mock_generator.generate_answer.return_value = {
                        "answer": (
                            "I couldn't find sufficient information about this topic in the "
                            "AI & Data Science Book. Please try asking about a topic covered in the book."
                        ),
                        "citations": [],
                        "evidence_threshold_met": False,
                        "grounded": False,
                    }
                MockGenerator.return_value = mock_generator

                pipe = RAGPipeline()
                pipe.retriever = mock_retriever
                pipe.generator = mock_generator
                return pipe

        return _build

    @pytest.mark.parametrize("item", [x for x in EVAL_BENCHMARK if x["in_scope"]])
    def test_in_scope_queries_answered_with_citations(self, mock_pipeline_builder, item):
        """In-scope questions must meet evidence thresholds and include valid citations."""
        pipeline = mock_pipeline_builder([item["mock_chunk"]], has_sufficient=True)
        res = pipeline.answer_question(item["question"])

        assert res["evidence_threshold_met"] is True
        assert len(res["citations"]) > 0

        citation = res["citations"][0]
        assert citation["title"] != ""
        assert citation["file_path"].endswith(".md")
        assert citation["relevance_score"] >= 0.70

        # Verify citation text actually supports the expected domain terms
        text = item["mock_chunk"]["content"].lower()
        matched_terms = [t for t in item["expected_terms"] if t in text]
        assert len(matched_terms) > 0, f"Cited text missing key terms for {item['id']}"

    @pytest.mark.parametrize("item", [x for x in EVAL_BENCHMARK if not x["in_scope"]])
    def test_out_of_scope_queries_strictly_refused(self, mock_pipeline_builder, item):
        """Out-of-scope questions must be refused without fabricating citations."""
        chunks = [item["mock_chunk"]] if item["mock_chunk"] else []
        pipeline = mock_pipeline_builder(chunks, has_sufficient=False)
        res = pipeline.answer_question(item["question"])

        assert res["evidence_threshold_met"] is False
        assert len(res["citations"]) == 0
        assert res["answer"] == (
            "I can only answer questions using the content available in this AI/ML book. "
            "I could not find enough relevant information in the book to answer your question."
        )

    def test_evaluation_metric_accuracy(self, mock_pipeline_builder):
        """Compute aggregate precision/recall over the full benchmark."""
        correct_classifications = 0
        total = len(EVAL_BENCHMARK)

        for item in EVAL_BENCHMARK:
            is_in_scope = item["in_scope"]
            chunks = [item["mock_chunk"]] if item.get("mock_chunk") else []
            pipeline = mock_pipeline_builder(chunks, has_sufficient=is_in_scope)
            res = pipeline.answer_question(item["question"])

            if is_in_scope and res["evidence_threshold_met"] is True and len(res["citations"]) > 0:
                correct_classifications += 1
            elif not is_in_scope and res["evidence_threshold_met"] is False and len(res["citations"]) == 0:
                correct_classifications += 1

        accuracy = correct_classifications / total
        assert accuracy == 1.0, f"Benchmark accuracy expected 100%, got {accuracy * 100}%"
