"""Text chunker for book content ingestion.

Splits Markdown/MDX content into meaningful chunks with heading preservation
and configurable token overlap.
"""
from typing import List, Dict, Any, Optional
import re
import tiktoken

# Use the same tokenizer as the Gemini model if available
try:
    encoding = tiktoken.encoding_for_model("gemini-embedding-001")
except Exception:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        class _OfflineEncoding:
            """Small deterministic fallback that never downloads tokenizer data."""
            @staticmethod
            def encode(value: str):
                return re.findall(r"\w+|[^\w\s]", value, flags=re.UNICODE)

        encoding = _OfflineEncoding()


def chunk_by_tokens(
    content: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Split content into overlapping token-based chunks.

    Args:
        content: Raw text/content to split
        chunk_size: Target number of tokens per chunk
        chunk_overlap: Number of overlapping tokens between adjacent chunks
        metadata: Additional metadata to attach to each chunk (module, lesson, heading, etc.)

    Returns:
        List of chunk dicts with 'id', 'content', 'token_count', and merged metadata
    """
    if not content or not content.strip():
        return []

    # Split by lines first to preserve structure
    lines = content.split("\n")

    chunks: List[Dict[str, Any]] = []
    current_chunk_lines: List[str] = []
    current_token_count = 0
    current_metadata: Dict[str, Any] = metadata.copy() if metadata else {}

    for line in lines:
        line_tokens = len(encoding.encode(line))

        # If adding this line would exceed chunk_size and we already have content,
        # finalize the current chunk and start a new one
        if current_token_count + line_tokens > chunk_size and current_chunk_lines:
            # Apply overlap: keep last few lines from current chunk
            overlap_lines: List[str] = []
            overlap_token_count = 0
            for prev_line in reversed(current_chunk_lines):
                if overlap_token_count + len(encoding.encode(prev_line)) <= chunk_overlap:
                    overlap_lines.insert(0, prev_line)
                    overlap_token_count += len(encoding.encode(prev_line))
                else:
                    break

            chunk_content = "\n".join(current_chunk_lines)
            chunk_id = _generate_chunk_id(chunk_content, metadata)

            chunks.append({
                "id": chunk_id,
                "content": chunk_content,
                "token_count": current_token_count,
                "metadata": current_metadata.copy(),
            })

            # Start new chunk with overlap lines
            current_chunk_lines = overlap_lines
            current_token_count = overlap_token_count

        current_chunk_lines.append(line)
        current_token_count += line_tokens

    # Don't forget the last chunk
    if current_chunk_lines:
        chunk_content = "\n".join(current_chunk_lines)
        chunk_id = _generate_chunk_id(chunk_content, metadata)

        chunks.append({
            "id": chunk_id,
            "content": chunk_content,
            "token_count": current_token_count,
            "metadata": current_metadata.copy(),
        })

    # Attach the nearest section and its heading ancestry to each chunk. The
    # file-level heading supplies the parent when a chunk starts mid-chapter.
    for chunk in chunks:
        hierarchy = {}
        base_heading = (metadata or {}).get("heading", "")
        if base_heading.startswith("#"):
            level = len(base_heading) - len(base_heading.lstrip("#"))
            hierarchy[level] = base_heading.lstrip("#").strip()
        for line in chunk["content"].splitlines():
            match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if not match:
                continue
            level, title = len(match.group(1)), match.group(2).strip()
            hierarchy = {depth: value for depth, value in hierarchy.items() if depth < level}
            hierarchy[level] = title
        ordered = [hierarchy[level] for level in sorted(hierarchy)]
        if ordered:
            chunk["metadata"]["heading"] = ordered[-1]
            chunk["metadata"]["heading_hierarchy"] = ordered

    return chunks


def _generate_chunk_id(content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Generate a stable identifier for a chunk based on its content."""
    import hashlib
    hash_input = content
    if metadata:
        hash_input += str(metadata)
    return f"chunk_{hashlib.sha256(hash_input.encode()).hexdigest()[:16]}"
