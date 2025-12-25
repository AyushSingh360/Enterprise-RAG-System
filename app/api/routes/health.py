"""
Health check endpoint.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from app import __version__
from app.schemas.common import HealthResponse
from app.config import get_settings

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check health status of RAG system.",
)
async def health_check() -> HealthResponse:
    """
    Check health of system.
    
    Returns basic health status without calling external APIs
    to avoid unnecessary latency and API costs.
    """
    settings = get_settings()
    
    components = {
        "api": "healthy",
        "config": "healthy" if settings.openai_api_key_value else "unhealthy",
    }
    
    # Check if FAISS index directory exists
    if settings.faiss_index_path.exists():
        components["vector_store"] = "healthy"
    else:
        components["vector_store"] = "initializing"
    
    overall = "healthy" if all(v == "healthy" for v in components.values()) else "degraded"
    
    return HealthResponse(
        status=overall,
        version=__version__,
        timestamp=datetime.now(timezone.utc),
        components=components,
    )
