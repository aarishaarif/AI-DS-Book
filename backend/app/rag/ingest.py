"""Content ingestion pipeline for book content.

Discovers, extracts, and indexes book content into the vector store.
"""
import os
from typing import List, Dict, Any, Optional, Tuple
from .embeddings import GeminiEmbedder
from .store import QdrantStore
from .chunker import chunk_by_tokens
from ..rag import retrieve
import logging

logger = logging.getLogger(__name__)


class BookContentIndexer:
    """Indexes book content into the Qdrant vector store."""

    def __init__(
        self,
        embedder: GeminiEmbedder,
        store: QdrantStore,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        book_dir: str = "../frontend/docs",
        supported_extensions: List[str] = (".md", ".mdx"),
    ):
        self.embedder = embedder
        self.store = store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.book_dir = book_dir
        self.supported_extensions = supported_extensions

    def discover_files(self) -> List[str]:
        """Discover supported book content files deterministically."""
        files = []
        for root, _, filenames in os.walk(self.book_dir):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in self.supported_extensions):
                    files.append(os.path.join(root, filename))
        return sorted(files)

    def extract_metadata_from_path(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Extract document and section metadata from file path.

        Assumes path structure like:
        docs/<module>/<lesson>/<filename> or docs/<module>/<lesson.md>

        Args:
            file_path: Full file path
            filename: Just the file name

        Returns:
            Metadata dict with module, lesson, title, file_path, source
        """
        rel_path = os.path.relpath(file_path, self.book_dir)
        parts = rel_path.split(os.sep)

        # Remove file extension from last part
        name_no_ext = os.path.splitext(parts[-1])[0]

        metadata: Dict[str, Any] = {
            "file_path": rel_path,
            "source": rel_path,
            "title": name_no_ext.replace("-", " ").replace("_", " ").title(),
        }

        if len(parts) >= 1:
            metadata["module"] = parts[0].replace("-", " ").replace("_", " ").title()
        if len(parts) >= 2:
            metadata["lesson"] = parts[1].replace("-", " ").replace("_", " ").title()

        return metadata

    def parse_markdown(self, file_path: str) -> Tuple[str, Dict[str, Any], Optional[str]]:
        """Parse a Markdown file into content and metadata.

        Args:
            file_path: Path to the Markdown file

        Returns:
            Tuple of (clean_text, metadata, heading)
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")

        # Extract title from frontmatter or first heading
        first_heading = None
        title = "Untitled"
        frontmatter = {}

        if lines and lines[0].startswith("---"):
            # Parse YAML frontmatter
            end_idx = next(
                i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"
            )
            for line in lines[1:end_idx]:
                if ":" in line:
                    key, value = line.split(":", 1)
                    frontmatter[key.strip()] = value.strip().strip('"')
            start_idx = end_idx + 1
        else:
            start_idx = 0

        # Find first heading
        for line in lines[start_idx:]:
            if line.startswith("#"):
                first_heading = line.strip()
                break

        # Build metadata from frontmatter or heading
        if frontmatter.get("title"):
            title = frontmatter["title"]
        elif first_heading:
            title = first_heading.lstrip("#").strip().strip("-")

        # Extract content after frontmatter, preserving headings
        return "\n".join(lines[start_idx:]), {"heading": first_heading or ""}, None

    def ingest_file(self, file_path: str) -> int:
        """Ingest a single file into the vector store.

        Args:
            file_path: Path to the Markdown file to ingest

        Returns:
            Number of chunks indexed
        """
        metadata = self.extract_metadata_from_path(file_path, os.path.basename(file_path))

        content, parsed_metadata, _ = self.parse_markdown(file_path)

        full_metadata = {**metadata, **parsed_metadata}
        full_metadata["title"] = metadata["title"]

        # Chunk the content
        chunks = chunk_by_tokens(
            content,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            metadata=full_metadata,
        )

        # Embed and store chunks
        indexed = 0
        for chunk in chunks:
            try:
                chunk_with_embedding = self.embedder.embed_chunk_with_metadata(chunk)
                self.store.index_chunk(chunk_with_embedding)
                indexed += 1
            except Exception as e:
                logger.warning(
                    f"Failed to index chunk {chunk.get('id')} from {file_path}: {e}"
                )

        return indexed

    def ingest_all(self, force_rebuild: bool = False) -> Dict[str, int]:
        """Ingest all discovered book content into the vector store.

        Args:
            force_rebuild: If True, clear the collection before indexing

        Returns:
            Dict with statistics about indexing
        """
        files = self.discover_files()
        logger.info(f"Discovered {len(files)} files in {self.book_dir}")

        stats = {"files": len(files), "chunks_indexed": 0, "errors": 0}

        try:
            if force_rebuild:
                self.store.clear_collection()
                logger.info("Cleared existing collection for rebuild")
        except Exception as e:
            logger.warning(f"Failed to clear collection: {e}")

        for file_path in files:
            try:
                n_chunks = self.ingest_file(file_path)
                stats["chunks_indexed"] += n_chunks
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
                stats["errors"] += 1

        logger.info(
            f"Indexing complete: {stats['chunks_indexed']} chunks from {stats['files']} files, {stats['errors']} errors"
        )

        return stats