"""
Tests for FastAPI endpoints.

These tests use TestClient with mocked RAG pipeline dependencies so they
run without real API credentials or a running Qdrant instance.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_mock_pipeline(answer: str = "Python is a programming language.", grounded: bool = True):
    """Build a mock RAGPipeline that returns a deterministic answer."""
    mock = MagicMock()
    mock.answer_question.return_value = {
        "answer": answer,
        "citations": [
            {
                "title": "Python Foundations",
                "module": "Module 1",
                "section": "Introduction",
                "file_path": "module-01-python-foundations.md",
                "relevance_score": 0.92,
            }
        ],
        "evidence_threshold_met": grounded,
        "grounded": grounded,
    }
    mock.ingest_content.return_value = {
        "files": 3,
        "chunks_indexed": 45,
        "errors": 0,
    }
    mock.get_collection_stats.return_value = {"total_chunks": 45}
    return mock


# ── Health endpoint ────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def setup_method(self):
        from app.main import app
        self.client = TestClient(app)

    def test_health_returns_200(self):
        response = self.client.get("/api/health")
        assert response.status_code == 200

    def test_health_returns_status_healthy(self):
        response = self.client.get("/api/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_returns_service_name(self):
        response = self.client.get("/api/health")
        data = response.json()
        assert "service" in data


# ── Chat endpoint ──────────────────────────────────────────────────────────────

class TestChatEndpoint:
    def setup_method(self):
        from app.main import app
        self.client = TestClient(app)
        self.mock_pipeline = make_mock_pipeline()

    def test_chat_returns_200_with_valid_question(self):
        with patch("app.main._pipeline", self.mock_pipeline):
            response = self.client.post(
                "/api/chat",
                json={"question": "What is Python?"},
            )
        assert response.status_code == 200

    def test_chat_response_has_required_fields(self):
        with patch("app.main._pipeline", self.mock_pipeline):
            response = self.client.post(
                "/api/chat",
                json={"question": "What is NumPy?"},
            )
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "evidence_threshold_met" in data

    def test_chat_answer_is_non_empty_string(self):
        with patch("app.main._pipeline", self.mock_pipeline):
            response = self.client.post(
                "/api/chat",
                json={"question": "What is pandas?"},
            )
        data = response.json()
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_chat_citations_are_list(self):
        with patch("app.main._pipeline", self.mock_pipeline):
            response = self.client.post(
                "/api/chat",
                json={"question": "Explain machine learning"},
            )
        data = response.json()
        assert isinstance(data["citations"], list)

    def test_chat_empty_question_returns_422(self):
        with patch("app.main._pipeline", self.mock_pipeline):
            response = self.client.post(
                "/api/chat",
                json={"question": ""},
            )
        assert response.status_code == 422

    def test_chat_missing_question_returns_422(self):
        with patch("app.main._pipeline", self.mock_pipeline):
            response = self.client.post("/api/chat", json={})
        assert response.status_code == 422

    def test_chat_whitespace_only_question_returns_422(self):
        with patch("app.main._pipeline", self.mock_pipeline):
            response = self.client.post(
                "/api/chat",
                json={"question": "   "},
            )
        assert response.status_code == 422

    def test_chat_pipeline_not_initialized_returns_503(self):
        """When pipeline is None, /api/chat should return 503."""
        with patch("app.main._pipeline", None):
            response = self.client.post(
                "/api/chat",
                json={"question": "What is Python?"},
            )
        assert response.status_code == 503

    def test_chat_unsupported_question_returns_refusal(self):
        """When the RAG pipeline returns no evidence, response indicates it."""
        mock_pipeline = make_mock_pipeline(
            answer=(
                "I couldn't find sufficient information about this topic "
                "in the AI & Data Science Book."
            ),
            grounded=False,
        )
        with patch("app.main._pipeline", mock_pipeline):
            response = self.client.post(
                "/api/chat",
                json={"question": "What is the capital of France?"},
            )
        data = response.json()
        assert data["evidence_threshold_met"] is False

    def test_chat_with_optional_session_id(self):
        with patch("app.main._pipeline", self.mock_pipeline):
            response = self.client.post(
                "/api/chat",
                json={"question": "What is deep learning?", "session_id": "session-abc"},
            )
        assert response.status_code == 200

    def test_chat_very_long_question(self):
        long_question = "What is Python? " * 100  # 1600 chars
        with patch("app.main._pipeline", self.mock_pipeline):
            response = self.client.post(
                "/api/chat",
                json={"question": long_question},
            )
        # Should still respond — not crash
        assert response.status_code in (200, 422)


# ── Ingest endpoint ────────────────────────────────────────────────────────────

class TestIngestEndpoint:
    def setup_method(self):
        from app.main import app
        import app.main as main_module
        self.client = TestClient(app)
        self.mock_pipeline = make_mock_pipeline()
        # Set directly on the module so get_rag_pipeline() returns our mock
        self._saved_pipeline = main_module._pipeline
        main_module._pipeline = self.mock_pipeline

    def teardown_method(self):
        import app.main as main_module
        main_module._pipeline = self._saved_pipeline

    def test_ingest_starts_background_job(self):
        with patch("app.routes.ingest._ingest_status",
                   {"running": False, "last_result": None, "last_error": None}), \
             patch("app.routes.ingest._run_ingest"):
            response = self.client.post("/api/ingest", json={"force_rebuild": False})
        assert response.status_code == 200
        assert response.json()["status"] == "started"

    def test_ingest_status_endpoint_exists(self):
        response = self.client.get("/api/ingest/status")
        assert response.status_code == 200

    def test_ingest_status_has_running_field(self):
        response = self.client.get("/api/ingest/status")
        data = response.json()
        assert "running" in data

    def test_ingest_pipeline_not_initialized_returns_503(self):
        with patch("app.main._pipeline", None), \
             patch("app.routes.ingest._ingest_status", {"running": False, "last_result": None, "last_error": None}):
            response = self.client.post("/api/ingest", json={"force_rebuild": False})
        assert response.status_code == 503


# ── Root endpoint ──────────────────────────────────────────────────────────────

class TestRootEndpoint:
    def setup_method(self):
        from app.main import app
        self.client = TestClient(app)

    def test_root_returns_200(self):
        response = self.client.get("/")
        assert response.status_code == 200

    def test_root_returns_message(self):
        response = self.client.get("/")
        data = response.json()
        assert "message" in data
