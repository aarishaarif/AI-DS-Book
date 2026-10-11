"""
Tests for the content chunker.

These tests run with no external services (no API, no Qdrant).
"""
import pytest
from app.rag.chunker import chunk_by_tokens, _generate_chunk_id


# ── chunk_by_tokens ────────────────────────────────────────────────────────────

class TestChunkByTokens:
    def test_empty_content_returns_empty_list(self):
        assert chunk_by_tokens("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_by_tokens("   \n\n   ") == []

    def test_short_content_produces_one_chunk(self):
        content = "Hello world."
        chunks = chunk_by_tokens(content, chunk_size=800)
        assert len(chunks) == 1

    def test_chunk_has_required_keys(self):
        chunks = chunk_by_tokens("Hello world.", chunk_size=800)
        chunk = chunks[0]
        assert "id" in chunk
        assert "content" in chunk
        assert "token_count" in chunk
        assert "metadata" in chunk

    def test_chunk_content_is_non_empty(self):
        chunks = chunk_by_tokens("Some content here.", chunk_size=800)
        for chunk in chunks:
            assert len(chunk["content"].strip()) > 0

    def test_large_content_splits_into_multiple_chunks(self, sample_markdown_content):
        chunks = chunk_by_tokens(sample_markdown_content, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1

    def test_chunk_token_count_within_limit(self, sample_markdown_content):
        chunk_size = 80
        chunks = chunk_by_tokens(sample_markdown_content, chunk_size=chunk_size, chunk_overlap=10)
        # Each chunk should be close to but not massively exceed the limit
        # (overlap may cause small exceedance)
        for chunk in chunks:
            assert chunk["token_count"] <= chunk_size + 20, (
                f"Chunk exceeds limit: {chunk['token_count']} tokens"
            )

    def test_metadata_is_attached_to_every_chunk(self):
        meta = {"module": "Module 1", "file_path": "test.md"}
        chunks = chunk_by_tokens("Short content.", metadata=meta)
        for chunk in chunks:
            assert chunk["metadata"]["module"] == "Module 1"
            assert chunk["metadata"]["file_path"] == "test.md"

    def test_metadata_is_copied_not_shared(self):
        """Each chunk should have an independent copy of metadata."""
        meta = {"module": "Module 1"}
        chunks = chunk_by_tokens("Line one.\n" * 50, chunk_size=30, chunk_overlap=5, metadata=meta)
        # Mutating one chunk's metadata should not affect others
        if len(chunks) > 1:
            chunks[0]["metadata"]["module"] = "CHANGED"
            assert chunks[1]["metadata"]["module"] == "Module 1"

    def test_chunk_ids_are_unique(self, sample_markdown_content):
        chunks = chunk_by_tokens(sample_markdown_content, chunk_size=50, chunk_overlap=5)
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_ids_are_stable(self):
        """Same content should always produce the same chunk ID."""
        content = "Python is a programming language."
        chunks1 = chunk_by_tokens(content)
        chunks2 = chunk_by_tokens(content)
        assert chunks1[0]["id"] == chunks2[0]["id"]

    def test_different_content_different_ids(self):
        id1 = chunk_by_tokens("Content A")[0]["id"]
        id2 = chunk_by_tokens("Content B")[0]["id"]
        assert id1 != id2

    def test_overlap_tokens_approximately_respected(self):
        """With overlap=20, consecutive chunks should share some tokens."""
        content = " ".join([f"word{i}" for i in range(200)])
        chunks = chunk_by_tokens(content, chunk_size=50, chunk_overlap=15)
        if len(chunks) >= 2:
            # End of chunk[0] and start of chunk[1] should share some words
            end_of_first = chunks[0]["content"].split()[-5:]
            start_of_second = chunks[1]["content"].split()[:5]
            shared = set(end_of_first) & set(start_of_second)
            assert len(shared) > 0, "Chunks should have overlapping tokens"

    def test_chunk_preserves_markdown_headings(self):
        content = "# Heading\n\nSome paragraph text.\n\n## Subheading\n\nMore text."
        chunks = chunk_by_tokens(content, chunk_size=100)
        full_text = "\n".join(c["content"] for c in chunks)
        assert "# Heading" in full_text
        assert "## Subheading" in full_text

    def test_no_metadata_produces_empty_metadata_dict(self):
        chunks = chunk_by_tokens("Hello.", metadata=None)
        assert chunks[0]["metadata"] == {}

    def test_returns_list(self):
        result = chunk_by_tokens("test content")
        assert isinstance(result, list)


# ── _generate_chunk_id ─────────────────────────────────────────────────────────

class TestGenerateChunkId:
    def test_returns_string(self):
        assert isinstance(_generate_chunk_id("content"), str)

    def test_starts_with_chunk_prefix(self):
        cid = _generate_chunk_id("content")
        assert cid.startswith("chunk_")

    def test_deterministic_without_metadata(self):
        assert _generate_chunk_id("hello") == _generate_chunk_id("hello")

    def test_deterministic_with_metadata(self):
        meta = {"module": "M1"}
        id1 = _generate_chunk_id("hello", meta)
        id2 = _generate_chunk_id("hello", meta)
        assert id1 == id2

    def test_different_content_different_ids(self):
        assert _generate_chunk_id("abc") != _generate_chunk_id("xyz")

    def test_same_content_different_metadata_different_ids(self):
        id1 = _generate_chunk_id("content", {"module": "M1"})
        id2 = _generate_chunk_id("content", {"module": "M2"})
        assert id1 != id2

    def test_id_length_is_fixed(self):
        """ID should be 'chunk_' + 16 hex chars = 22 chars."""
        cid = _generate_chunk_id("test")
        assert len(cid) == len("chunk_") + 16
