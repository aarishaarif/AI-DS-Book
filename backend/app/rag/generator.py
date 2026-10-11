"""Answer generation for RAG pipeline.

Constructs prompts and generates answers from retrieved book content
using the Gemini generative model, ensuring answers are grounded in
retrieved evidence only.
"""
import google.genai as genai
from google.genai import types
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generates answers from retrieved document chunks using Gemini."""

    def __init__(
        self,
        generation_model: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
        evidence_threshold: float = 0.7,
    ):
        self.generation_model = generation_model
        self.api_key = api_key
        self.evidence_threshold = evidence_threshold

        if api_key:
            genai.configure(api_key=api_key)

    def _get_client(self):
        """Get or create Gemini client."""
        return genai

    def _build_prompt(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        citation_style: str = "clickable",
    ) -> str:
        """Build a grounded prompt for answer generation.

        Args:
            question: User's question
            chunks: Retrieved chunks with evidence
            citation_style: How to format citations in the prompt

        Returns:
            Constructed prompt string
        """
        # Build context from retrieved chunks with citations
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            source_info = metadata.get("source", metadata.get("file_path", "unknown"))
            section = metadata.get("heading", metadata.get("section", "unknown section"))
            module = metadata.get("module", "")

            context_block = f"""[{i}] Source: {source_info}
Section: {section}"""
            if module:
                context_block += f" (Module: {module})"
            context_block += f"""

Content:
{chunk['content']}"""

            context_parts.append(context_block)

        context = "\n\n".join(context_parts)

        prompt = f"""You are an AI assistant for the AI & Data Science Learning Book.
Answer the user's question using ONLY the provided book content below.

IMPORTANT RULES:
1. Answer strictly from the book content - do not use general knowledge
2. If the book does not contain sufficient evidence to answer, respond:
   "I couldn't find sufficient information about this topic in the AI & Data Science Book. Please try asking about a topic covered in the book."
3. Provide clickable citations using the [N] format that reference the source sections
4. Do not fabricate citations, lesson titles, or source URLs
5. If citing code examples, include the exact code from the book

Book content for this question:
{context}

User question: {question}

Answer:"""

        return prompt

    def generate_answer(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        citation_style: str = "clickable",
    ) -> Dict[str, Any]:
        """Generate an answer from retrieved chunks.

        Args:
            question: User's question
            chunks: Retrieved chunks with evidence
            citation_style: How to format citations in output

        Returns:
            Dict with answer, citations, and evidence status
        """
        try:
            # Check if sufficient evidence exists
            if not chunks:
                return {
                    "answer": (
                        "I couldn't find sufficient information about this topic in the AI & Data Science Book. "
                        "Please try asking about a topic covered in the book."
                    ),
                    "citations": [],
                    "evidence_threshold_met": False,
                    "grounded": False,
                }

            # Check evidence threshold
            total_score = sum(chunk.get("score", 0.0) for chunk in chunks)
            avg_score = total_score / len(chunks) if chunks else 0.0
            evidence_strong = avg_score >= self.evidence_threshold

            # Build prompt and generate
            prompt = self._build_prompt(question, chunks, citation_style)
            client = self._get_client()

            # Generate using Gemini
            model_name = self.generation_model or "gemini-2.0-flash-exp"
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=1024,
                ),
            )

            answer_text = response.text if hasattr(response, 'text') else str(response)

            # Format citations
            citations = self._format_citations_from_chunks(chunks)

            return {
                "answer": answer_text,
                "citations": citations,
                "evidence_threshold_met": evidence_strong,
                "grounded": True,
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": (
                    "I encountered an error while generating the answer. "
                    "Please try again or ask about a different topic."
                ),
                "citations": [],
                "evidence_threshold_met": False,
                "grounded": False,
                "error": str(e),
            }

    def _format_citations_from_chunks(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format citations from retrieved chunks."""
        citations = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            citation = {
                "source_title": metadata.get("title", "Untitled"),
                "source_section": metadata.get("heading", metadata.get("section", "")),
                "source_module": metadata.get("module", ""),
                "source_file": metadata.get("file_path", ""),
                "chunk_id": chunk.get("id", ""),
                "relevance_score": chunk.get("score", 0.0),
                "citation_number": i,
            }
            citations.append(citation)
        return citations