#!/usr/bin/env python3
"""
CLI tool for repeatable book content ingestion into Qdrant.

Usage:
    python ingest_cli.py [--force-rebuild] [--book-dir PATH] [--dry-run]
"""
import sys
import os
import argparse
import logging
from typing import Dict, Any

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import get_settings
from app.rag.chunker import chunk_by_tokens
from app.rag.ingest import BookContentIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest-cli")


def run_dry_run(book_dir: str, chunk_size: int = 800, chunk_overlap: int = 100) -> int:
    """Validate and chunk files without calling external APIs."""
    logger.info(f"Running in DRY-RUN mode against: {book_dir}")
    if not os.path.exists(book_dir):
        logger.error(f"Book directory not found: {book_dir}")
        return 1

    files = []
    for root, _, filenames in os.walk(book_dir):
        for f in filenames:
            if f.endswith((".md", ".mdx")):
                files.append(os.path.join(root, f))
    files.sort()

    if not files:
        logger.error(f"No Markdown/MDX files discovered in {book_dir}")
        return 1

    logger.info(f"Discovered {len(files)} content files:")
    total_chunks = 0
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            content = fp.read()
        chunks = chunk_by_tokens(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        total_chunks += len(chunks)
        rel = os.path.relpath(f, book_dir)
        logger.info(f"  • {rel} -> {len(chunks)} chunks")

    logger.info(f"Dry-run complete. Total files: {len(files)}, Total chunks: {total_chunks}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest AI & Data Science Book content into Qdrant vector database."
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Clear existing Qdrant collection and re-index from scratch.",
    )
    parser.add_argument(
        "--book-dir",
        type=str,
        default=None,
        help="Directory containing Markdown files (defaults to BOOK_DIR from settings).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk files locally without calling Gemini or Qdrant.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check Qdrant connectivity and collection configuration without modifying data.",
    )

    args = parser.parse_args()
    settings = get_settings()

    book_dir = args.book_dir or settings.book_dir
    # Resolve relative path relative to backend dir
    if not os.path.isabs(book_dir):
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(backend_dir, book_dir)
        if os.path.exists(candidate):
            book_dir = candidate
        else:
            # Check frontend/docs
            alt = os.path.abspath(os.path.join(backend_dir, "..", "frontend", "docs"))
            if os.path.exists(alt):
                book_dir = alt

    logger.info(f"Target book content directory: {book_dir}")

    if args.check:
        try:
            from qdrant_client import QdrantClient

            if not settings.qdrant_url.strip():
                logger.error(
                    "Qdrant check failed: QDRANT_URL is not configured. "
                    "Set it in backend/.env or the backend process environment."
                )
                return 1

            # Use the raw client for this read-only check. QdrantStore is the
            # ingestion path and may create a missing collection by design.
            client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
            client.get_collections()
            info = client.get_collection(collection_name=settings.qdrant_collection_name)
            configured_size = getattr(getattr(info, "config", None), "params", None)
            logger.info(f"Qdrant connection: OK; collection={settings.qdrant_collection_name}; points={info.points_count}")
            logger.info(f"Configured embedding dimension: {settings.embedding_dimension}; collection params={configured_size}")
            return 0
        except Exception as e:
            logger.error(f"Qdrant check failed: {e}")
            return 1

    if args.dry_run:
        return run_dry_run(book_dir, settings.chunk_size, settings.chunk_overlap)

    # Validate prerequisites for live ingestion
    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error(
            "GEMINI_API_KEY is not configured! "
            "Please set GEMINI_API_KEY in backend/.env or your environment."
        )
        logger.info("Tip: You can run with '--dry-run' to test chunking without API credentials.")
        return 1

    try:
        from app.rag.pipeline import RAGPipeline

        logger.info("Initializing RAG pipeline components...")
        pipeline = RAGPipeline(
            embedding_model=settings.gemini_embedding_model,
            generation_model=settings.gemini_generation_model,
            api_key=settings.gemini_api_key or settings.gemini_embedding_api_key,
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key,
            collection_name=settings.qdrant_collection_name,
            embedding_dimension=settings.embedding_dimension,
            book_dir=book_dir,
            evidence_threshold=settings.evidence_threshold,
            retrieval_limit=settings.retrieval_limit,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        logger.info(f"Starting ingestion (force_rebuild={args.force_rebuild})...")
        stats = pipeline.ingest_content(force_rebuild=args.force_rebuild)

        if stats.get("errors", 0) > 0 and stats.get("chunks_indexed", 0) == 0:
            logger.error(f"Ingestion failed with errors: {stats}")
            return 1

        logger.info("=" * 60)
        logger.info("INGESTION SUCCESSFUL")
        logger.info(f"Files discovered: {stats.get('files', 0)}")
        logger.info(f"Chunks indexed:   {stats.get('chunks_indexed', 0)}")
        logger.info(f"Errors:           {stats.get('errors', 0)}")
        logger.info("=" * 60)
        return 0

    except Exception as e:
        logger.error(f"Fatal error during ingestion: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
