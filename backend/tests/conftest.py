"""Shared pytest fixtures for the AI & Data Science Learning Book backend tests."""
import pytest
import os
import sys

# Ensure the backend/app package is importable when running tests from backend/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def sample_markdown_content() -> str:
    """A realistic Markdown document for chunking and ingestion tests."""
    return """---
title: Test Module
sidebar_position: 1
---

# Introduction to Testing

This module covers automated testing in Python.

## Why Test?

Tests catch bugs early and give confidence when refactoring.

### Unit Tests

Unit tests verify individual functions in isolation.

```python
def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5
```

### Integration Tests

Integration tests verify that multiple components work together.

## Test Frameworks

Python has several testing frameworks:
- pytest (recommended)
- unittest (standard library)
- nose2

## Writing Good Tests

A good test is:
1. Fast
2. Isolated
3. Deterministic
4. Readable

## Practice

Write a test for a function that checks if a number is prime.
"""


@pytest.fixture
def mock_embedding() -> list:
    """A mock 768-dimensional embedding vector."""
    import random
    random.seed(42)
    return [random.uniform(-1, 1) for _ in range(768)]


@pytest.fixture
def sample_chunk(mock_embedding) -> dict:
    """A sample chunk with embedding, ready for vector store operations."""
    return {
        "id": "chunk_abc123",
        "content": "Python is a high-level programming language.",
        "token_count": 9,
        "metadata": {
            "title": "Python Foundations",
            "module": "Module 1",
            "lesson": "Introduction",
            "file_path": "module-01-python-foundations.md",
            "source": "module-01-python-foundations.md",
            "heading": "# Python Foundations",
        },
        "embedding": mock_embedding,
    }
