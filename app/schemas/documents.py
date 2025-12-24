"""
Document-related schemas for ingestion operations.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Any
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    """Supported document types for ingestion."""
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    SQL = "sql"


class DocumentMetadata(BaseModel):
    """Metadata associated with a document."""
    
    source: str = Field(
        description="Original source/filename of the document"
    )
    document_type: DocumentType = Field(
        description="Type of the document"
    )
    page_number: Optional[int] = Field(
        default=None,
        description="Page number if applicable"
    )
    section: Optional[str] = Field(
        default=None,
        description="Section or chapter name if applicable"
    )
    title: Optional[str] = Field(
        default=None,
        description="Document title if available"
    )
    author: Optional[str] = Field(
        default=None,
        description="Document author if available"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when document was ingested"
    )
    custom_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom metadata"
    )


class IngestRequest(BaseModel):
    """Request schema for document ingestion."""
    
    document_type: DocumentType = Field(
        description="Type of document being ingested"
    )
    content: Optional[str] = Field(
        default=None,
        description="Raw text content (for markdown/sql)"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Path to file on server (for PDF/DOCX)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata to attach"
    )
    
    # SQL-specific fields
    sql_query: Optional[str] = Field(
        default=None,
        description="SQL query to execute for data extraction"
    )
    connection_string: Optional[str] = Field(
        default=None,
        description="Database connection string (for SQL type)"
    )
    
    @field_validator("file_path", mode="before")
    @classmethod
    def validate_file_path(cls, v: Optional[str]) -> Optional[str]:
        """Validate that file path exists if provided."""
        if v is not None:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"File not found: {v}")
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "document_type": "pdf",
                "file_path": "/path/to/document.pdf",
                "metadata": {
                    "department": "Engineering",
                    "project": "RAG System"
                }
            }
        }
    }


class DocumentInfo(BaseModel):
    """Information about an ingested document."""
    
    document_id: str = Field(
        description="Unique identifier for the document"
    )
    source: str = Field(
        description="Original source of the document"
    )
    document_type: DocumentType = Field(
        description="Type of the document"
    )
    chunk_count: int = Field(
        description="Number of chunks created from this document"
    )
    ingested_at: datetime = Field(
        description="Timestamp when document was ingested"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata"
    )


class IngestResponse(BaseModel):
    """Response schema for document ingestion."""
    
    success: bool = Field(
        description="Whether ingestion was successful"
    )
    message: str = Field(
        description="Status message"
    )
    documents: list[DocumentInfo] = Field(
        default_factory=list,
        description="List of ingested documents with their info"
    )
    total_chunks: int = Field(
        default=0,
        description="Total number of chunks created"
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "message": "Successfully ingested 1 document",
                "documents": [
                    {
                        "document_id": "doc_abc123",
                        "source": "technical_manual.pdf",
                        "document_type": "pdf",
                        "chunk_count": 45,
                        "ingested_at": "2024-01-15T10:30:00Z",
                        "metadata": {"department": "Engineering"}
                    }
                ],
                "total_chunks": 45,
                "processing_time_ms": 1523.45
            }
        }
    }
