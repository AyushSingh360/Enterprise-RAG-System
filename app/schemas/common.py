"""
Common schemas used across the application.
"""

from datetime import datetime
from typing import Optional, Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str = Field(
        default="healthy",
        description="Service health status"
    )
    version: str = Field(
        description="Application version"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Individual component health statuses"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00Z",
                "components": {
                    "vector_store": "healthy",
                    "llm": "healthy",
                    "embeddings": "healthy"
                }
            }
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    
    error: str = Field(
        description="Error type or code"
    )
    message: str = Field(
        description="Human-readable error message"
    )
    details: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "ValidationError",
                "message": "Invalid document format",
                "details": {"field": "file_type", "value": "xyz"},
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    }
