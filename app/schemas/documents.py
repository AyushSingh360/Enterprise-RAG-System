"""
Document and ingestion schemas.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any

from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"


class IngestRequest(BaseModel):
    """Request schema for document ingestion."""
    
    document_type: DocumentType = Field(
        description="Type of document to ingest"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Path to file (required for PDF/DOCX)"
    )
    content: Optional[str] = Field(
        default=None,
        description="Raw content (for Markdown)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata to attach"
    )
    
    @field_validator("file_path", mode="after")
    @classmethod
    def validate_file_path(cls, v: Optional[str], info) -> Optional[str]:
        """Validate file path is provided for file-based types."""
        return v
    
    def model_post_init(self, __context: Any) -> None:
        """Validate that required fields are present based on type."""
        if self.document_type in (DocumentType.PDF, DocumentType.DOCX):
            if not self.file_path:
                raise ValueError(
                    f"{self.document_type.value.upper()} requires file_path"
                )
        elif self.document_type == DocumentType.MARKDOWN:
            if not self.content and not self.file_path:
                raise ValueError("Markdown requires content or file_path")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "document_type": "pdf",
                    "file_path": "/path/to/document.pdf",
                    "metadata": {"department": "Engineering"}
                },
                {
                    "document_type": "markdown",
                    "content": "# Title\n\nDocument content here.",
                    "metadata": {"source": "manual_input"}
                }
            ]
        }
    }


class DocumentInfo(BaseModel):
    """Information about an ingested document."""
    
    document_id: str = Field(
        description="Unique document identifier"
    )
    source: str = Field(
        description="Document source name"
    )
    document_type: DocumentType = Field(
        description="Type of document"
    )
    chunk_count: int = Field(
        ge=0,
        description="Number of chunks created"
    )
    ingested_at: datetime = Field(
        description="Ingestion timestamp (UTC)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "document_id": "doc_abc123def456",
                "source": "policy_manual.pdf",
                "document_type": "pdf",
                "chunk_count": 42,
                "ingested_at": "2024-01-15T10:30:00Z",
                "metadata": {"department": "HR"}
            }
        }
    }


class IngestResponse(BaseModel):
    """Response schema for document ingestion."""
    
    success: bool = Field(
        description="Whether ingestion succeeded"
    )
    message: str = Field(
        description="Status message"
    )
    documents: list[DocumentInfo] = Field(
        default_factory=list,
        description="Ingested document information"
    )
    total_chunks: int = Field(
        default=0,
        ge=0,
        description="Total chunks created"
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Processing time in milliseconds"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "message": "Successfully ingested with 42 chunks",
                "documents": [
                    {
                        "document_id": "doc_abc123def456",
                        "source": "policy_manual.pdf",
                        "document_type": "pdf",
                        "chunk_count": 42,
                        "ingested_at": "2024-01-15T10:30:00Z",
                        "metadata": {}
                    }
                ],
                "total_chunks": 42,
                "processing_time_ms": 1523.45
            }
        }
    }
