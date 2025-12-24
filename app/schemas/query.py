"""
Query-related schemas for retrieval and generation operations.
"""

from datetime import datetime
from typing import Optional, Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request schema for querying the RAG system."""
    
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The question to answer from the document corpus"
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Number of documents to retrieve (overrides default)"
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold (overrides default)"
    )
    include_context: bool = Field(
        default=False,
        description="Whether to include retrieved context in response"
    )
    metadata_filter: Optional[dict[str, Any]] = Field(
        default=None,
        description="Filter documents by metadata fields"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is the company's policy on remote work?",
                "top_k": 5,
                "similarity_threshold": 0.7,
                "include_context": True,
                "metadata_filter": {"department": "HR"}
            }
        }
    }


class SourceCitation(BaseModel):
    """Citation information for a source document."""
    
    document_id: str = Field(
        description="Unique identifier of the source document"
    )
    source: str = Field(
        description="Original source name/path"
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
        description="Similarity/relevance score"
    )
    chunk_text: Optional[str] = Field(
        default=None,
        description="Relevant text excerpt from the source"
    )


class RetrievedContext(BaseModel):
    """Context retrieved from the vector store."""
    
    chunk_id: str = Field(
        description="Unique identifier of the chunk"
    )
    document_id: str = Field(
        description="Parent document identifier"
    )
    text: str = Field(
        description="The retrieved text content"
    )
    similarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Similarity score to the query"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata"
    )


class QueryResponse(BaseModel):
    """Response schema for query operations."""
    
    success: bool = Field(
        description="Whether the query was processed successfully"
    )
    answer: str = Field(
        description="The generated answer or 'Answer not found in documents'"
    )
    sources: list[SourceCitation] = Field(
        default_factory=list,
        description="List of source citations for the answer"
    )
    context: Optional[list[RetrievedContext]] = Field(
        default=None,
        description="Retrieved context (if include_context=True)"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score of the answer"
    )
    query_time_ms: float = Field(
        default=0.0,
        description="Total query processing time in milliseconds"
    )
    retrieval_time_ms: float = Field(
        default=0.0,
        description="Time spent on retrieval in milliseconds"
    )
    generation_time_ms: float = Field(
        default=0.0,
        description="Time spent on generation in milliseconds"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "answer": "According to the company policy, employees can work remotely up to 3 days per week with manager approval.",
                "sources": [
                    {
                        "document_id": "doc_abc123",
                        "source": "hr_policies.pdf",
                        "page_number": 15,
                        "section": "Remote Work Policy",
                        "relevance_score": 0.92,
                        "chunk_text": "Employees may request remote work arrangements..."
                    }
                ],
                "context": None,
                "confidence": 0.89,
                "query_time_ms": 1245.67,
                "retrieval_time_ms": 89.23,
                "generation_time_ms": 1156.44
            }
        }
    }


class NoAnswerResponse(QueryResponse):
    """Special response when no relevant context is found."""
    
    answer: str = Field(
        default="Answer not found in documents",
        description="Fixed response for no context scenarios"
    )
    confidence: float = Field(
        default=0.0,
        description="Zero confidence when no answer found"
    )
    sources: list[SourceCitation] = Field(
        default_factory=list,
        description="Empty sources list"
    )
