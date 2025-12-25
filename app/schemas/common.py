"""
Common schemas used across the API.
"""

from datetime import datetime, timezone
from typing import Optional, Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(
        description="Overall health status: healthy, degraded, or unhealthy"
    )
    version: str = Field(
        description="Application version"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Health check timestamp (UTC)"
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
                    "embeddings": "healthy",
                    "vector_store": "healthy",
                    "llm": "healthy"
                }
            }
        }
    }


class ErrorDetail(BaseModel):
    """Detailed error information."""
    
    field: Optional[str] = Field(
        default=None,
        description="Field that caused the error"
    )
    message: str = Field(
        description="Error message"
    )
    code: Optional[str] = Field(
        default=None,
        description="Error code"
    )


class ErrorResponse(BaseModel):
    """Structured error response."""
    
    error: str = Field(
        description="Error type/category"
    )
    message: str = Field(
        description="Human-readable error message"
    )
    details: Optional[list[ErrorDetail]] = Field(
        default=None,
        description="Detailed error information"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp (UTC)"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "ValidationError",
                "message": "Request validation failed",
                "details": [
                    {"field": "question", "message": "Field required", "code": "missing"}
                ],
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "abc123"
            }
        }
    }
