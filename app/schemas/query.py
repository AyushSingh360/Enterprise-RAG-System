"""
Query and response schemas with mandatory citations.
"""

from typing import Optional, Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request schema for querying the RAG system."""
    
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question to answer from documents"
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Max documents to retrieve (default: 5)"
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Min similarity score (default: 0.7)"
    )
    include_context: bool = Field(
        default=False,
        description="Include retrieved context in response"
    )
    metadata_filter: Optional[dict[str, Any]] = Field(
        default=None,
        description="Filter by metadata fields"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is the company's vacation policy?",
                "top_k": 5,
                "similarity_threshold": 0.7,
                "include_context": False,
                "metadata_filter": None
            }
        }
    }


class SourceCitation(BaseModel):
    """
    Citation for a source document.
    
    MANDATORY in every response with an answer.
    """
    
    document_id: str = Field(
        description="Document identifier"
    )
    source: str = Field(
        description="Source name (filename)"
    )
    page_number: Optional[int] = Field(
        default=None,
        description="Page number if applicable"
    )
    section: Optional[str] = Field(
        default=None,
        description="Section name if applicable"
    )
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance/similarity score"
    )
    chunk_text: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Excerpt from source (max 500 chars)"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "document_id": "doc_abc123",
                "source": "hr_policies.pdf",
                "page_number": 15,
                "section": "Vacation Policy",
                "relevance_score": 0.92,
                "chunk_text": "Employees are entitled to 20 days..."
            }
        }
    }


class RetrievedContext(BaseModel):
    """Retrieved context chunk (optional in response)."""
    
    chunk_id: str = Field(
        description="Chunk identifier"
    )
    document_id: str = Field(
        description="Parent document identifier"
    )
    text: str = Field(
        description="Full chunk text"
    )
    similarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Similarity score"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata"
    )


class QueryResponse(BaseModel):
    """
    Response schema for queries.
    
    IMPORTANT: sources array is MANDATORY and contains citations.
    If no answer found, sources will be empty and answer will be
    "Answer not found in documents."
    """
    
    success: bool = Field(
        description="Whether query was processed successfully"
    )
    answer: str = Field(
        description="Generated answer or 'Answer not found in documents.'"
    )
    sources: list[SourceCitation] = Field(
        description="MANDATORY: Source citations for the answer (empty if no answer)"
    )
    context: Optional[list[RetrievedContext]] = Field(
        default=None,
        description="Retrieved context (only if include_context=True)"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Answer confidence (0.0 = no answer)"
    )
    query_time_ms: float = Field(
        ge=0,
        description="Total query time in milliseconds"
    )
    retrieval_time_ms: float = Field(
        ge=0,
        description="Retrieval phase time in milliseconds"
    )
    generation_time_ms: float = Field(
        ge=0,
        description="Generation phase time in milliseconds"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "answer": "Employees are entitled to 20 vacation days per year. [Source 1]",
                    "sources": [
                        {
                            "document_id": "doc_abc123",
                            "source": "hr_policies.pdf",
                            "page_number": 15,
                            "section": "Vacation Policy",
                            "relevance_score": 0.92,
                            "chunk_text": "Employees are entitled to 20 days..."
                        }
                    ],
                    "context": None,
                    "confidence": 0.89,
                    "query_time_ms": 1245.67,
                    "retrieval_time_ms": 89.23,
                    "generation_time_ms": 1156.44
                },
                {
                    "success": True,
                    "answer": "Answer not found in documents.",
                    "sources": [],
                    "context": None,
                    "confidence": 0.0,
                    "query_time_ms": 234.56,
                    "retrieval_time_ms": 234.56,
                    "generation_time_ms": 0.0
                }
            ]
        }
    }
