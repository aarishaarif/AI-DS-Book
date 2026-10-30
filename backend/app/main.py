"""
Backend API for AI & Data Science Learning Book.
Handles RAG chatbot, ingestion, and curriculum management.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

# Configuration
PROJECT_NAME = "AI & Data Science Learning Book"

# Import modules after app definition to avoid circular imports
from .routes.chat import router as chat_router
from .routes.ingest import router as ingest_router
from .routes.health import router as health_router
from .config import get_settings
from .rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Global pipeline instance (lazy-initialized)
_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global _pipeline
    settings = get_settings()

    if not (settings.gemini_api_key or settings.gemini_embedding_api_key):
        logger.warning("RAG pipeline is disabled: GEMINI_API_KEY is not configured")
        yield
        return

    try:
        _pipeline = RAGPipeline(
            embedding_model=settings.gemini_embedding_model,
            generation_model=settings.gemini_generation_model,
            api_key=settings.gemini_api_key or settings.gemini_embedding_api_key,
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key,
            collection_name=settings.qdrant_collection_name,
            embedding_dimension=settings.embedding_dimension,
            book_dir=settings.book_dir,
            evidence_threshold=settings.evidence_threshold,
            retrieval_limit=settings.retrieval_limit,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        logger.info("RAG pipeline initialized")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        # Don't fail startup if pipeline can't initialize - allow degraded functionality

    yield


app = FastAPI(
    title=PROJECT_NAME,
    version="0.1.0",
    description="Backend API for the AI & Data Science Learning Book with RAG chatbot",
    lifespan=lifespan,
)

# CORS - configure for intended frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix="/api", tags=["health"])
app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(ingest_router, prefix="/api", tags=["ingest"])


@app.get("/health", tags=["health"])
async def root_health():
    """Service health check endpoint."""
    return {"status": "healthy", "service": "AI & Data Science Learning Book API"}


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the AI & Data Science Learning Book API"}


def get_rag_pipeline() -> RAGPipeline:
    """Get the global RAG pipeline instance."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return _pipeline


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
