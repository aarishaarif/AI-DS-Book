"""Content ingestion and reindexing endpoints.

POST /api/ingest  — trigger full or incremental reindex of book content.
This route is wired to the global RAGPipeline instance from main.py.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])

# Simple in-memory job tracker so callers can check progress
_ingest_status: Dict[str, Any] = {
    "running": False,
    "last_result": None,
    "last_error": None,
}


class IngestRequest(BaseModel):
    """Request model for content ingestion."""
    force_rebuild: bool = Field(
        False,
        description="If true, clears the existing vector collection before indexing.",
    )


class IngestResponse(BaseModel):
    """Response model for ingestion results."""
    status: str = Field(..., description="'started', 'completed', or 'error'")
    message: str
    files_discovered: int = 0
    chunks_indexed: int = 0
    errors: int = 0


def _run_ingest(force_rebuild: bool) -> None:
    """Background task: runs the full ingestion pipeline."""
    global _ingest_status
    _ingest_status["running"] = True
    _ingest_status["last_error"] = None

    try:
        from ..main import get_rag_pipeline  # avoid circular import at module level
        pipeline = get_rag_pipeline()
        stats = pipeline.ingest_content(force_rebuild=force_rebuild)
        _ingest_status["last_result"] = stats
        logger.info(f"Ingestion complete: {stats}")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        _ingest_status["last_error"] = str(e)
    finally:
        _ingest_status["running"] = False


@router.post("", response_model=IngestResponse)
async def start_ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Trigger content ingestion (or full rebuild).

    This discovers all Markdown/MDX files in the book directory,
    chunks them, generates Gemini embeddings, and stores vectors in Qdrant.

    - `force_rebuild=false` — incremental: upserts changed chunks.
    - `force_rebuild=true`  — full rebuild: clears the collection first.

    The ingestion runs in the background. Check GET /api/ingest/status
    to monitor progress.
    """
    if _ingest_status["running"]:
        raise HTTPException(
            status_code=409,
            detail="An ingestion job is already running. Wait for it to finish.",
        )

    try:
        # Quick connectivity check before launching the background job
        from ..main import get_rag_pipeline
        get_rag_pipeline()  # raises 503 if pipeline not initialized
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"RAG pipeline unavailable: {e}")

    background_tasks.add_task(_run_ingest, request.force_rebuild)

    return IngestResponse(
        status="started",
        message=(
            "Ingestion started in background. "
            "Check GET /api/ingest/status for progress."
        ),
    )


@router.get("/status", response_model=Dict[str, Any])
async def ingest_status():
    """Return the current ingestion job status and result of the last run."""
    return {
        "running": _ingest_status["running"],
        "last_result": _ingest_status["last_result"],
        "last_error": _ingest_status["last_error"],
    }


@router.post("/sync", response_model=IngestResponse)
async def ingest_sync(request: IngestRequest):
    """
    Run ingestion synchronously (blocking).

    Useful for scripts and CI pipelines. Waits until ingestion completes
    before returning results.
    """
    if _ingest_status["running"]:
        raise HTTPException(
            status_code=409,
            detail="An ingestion job is already running.",
        )

    try:
        from ..main import get_rag_pipeline
        pipeline = get_rag_pipeline()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"RAG pipeline unavailable: {e}")

    _ingest_status["running"] = True
    try:
        stats = pipeline.ingest_content(force_rebuild=request.force_rebuild)
        _ingest_status["last_result"] = stats
        return IngestResponse(
            status="completed",
            message="Ingestion completed successfully.",
            files_discovered=stats.get("files", 0),
            chunks_indexed=stats.get("chunks_indexed", 0),
            errors=stats.get("errors", 0),
        )
    except Exception as e:
        logger.error(f"Sync ingestion failed: {e}")
        _ingest_status["last_error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        _ingest_status["running"] = False
