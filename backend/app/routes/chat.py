"""Chat RAG endpoint."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Request model for chat questions."""
    question: str = Field(..., description="User's question about the book content")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    include_citations: bool = Field(True, description="Whether to include source citations")

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v


class ChatResponse(BaseModel):
    """Response model for chat answers."""
    answer: str = Field(..., description="Answer grounded in book content")
    citations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Clickable source citations with title, section, and URL/path",
    )
    evidence_threshold_met: bool = Field(
        True, description="Whether sufficient evidence was found in the book"
    )
    grounded: bool = True
    classification: Optional[str] = None


@router.post("", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest):
    """Answer a question using RAG from the book content."""
    try:
        from ..main import get_rag_pipeline

        pipeline = get_rag_pipeline()
        result = pipeline.answer_question(request.question)

        return ChatResponse(
            answer=result["answer"],
            citations=result.get("citations", []),
            evidence_threshold_met=result.get("evidence_threshold_met", False),
            grounded=result.get("grounded", False),
            classification=result.get("classification"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat service error")
        return ChatResponse(
            answer="I'm temporarily unable to search the book. Please try again shortly.",
            citations=[], evidence_threshold_met=False, grounded=False,
            classification="service_error",
        )
