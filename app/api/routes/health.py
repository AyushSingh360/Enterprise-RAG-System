"""
Health check endpoint.
"""

from datetime import datetime
from fastapi import APIRouter, Depends

from app import __version__
from app.schemas.common import HealthResponse
from app.api.dependencies import get_retriever_service
from app.core.retriever import RetrieverService

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the RAG system and its components."
)
async def health_check(
    retriever: RetrieverService = Depends(get_retriever_service)
) -> HealthResponse:
    """
    Returns health status of all system components.
    """
    components = {}
    
    # Check embedding service
    embedding_health = await retriever.embedding_service.health_check()
    components["embeddings"] = embedding_health.get("status", "unknown")
    
    # Check vector store
    vector_health = await retriever.vector_store.health_check()
    components["vector_store"] = vector_health.get("status", "unknown")
    
    # Check LLM service
    llm_health = await retriever.llm_service.health_check()
    components["llm"] = llm_health.get("status", "unknown")
    
    # Overall status
    all_healthy = all(s == "healthy" for s in components.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version=__version__,
        timestamp=datetime.utcnow(),
        components=components
    )
