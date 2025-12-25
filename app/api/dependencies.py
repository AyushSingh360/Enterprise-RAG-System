"""
Dependency injection for API routes.

Provides singleton instances of all services.
Services are lazily initialized on first access.
"""

from typing import Optional
import structlog

from app.config import Settings, get_settings
from app.core.embeddings import EmbeddingService
from app.core.vector_store import VectorStoreService
from app.core.llm import LLMService
from app.core.retriever import RetrieverService
from app.ingestion.pipeline import IngestionPipeline, IngestionService

logger = structlog.get_logger(__name__)


# =============================================================================
# Singleton Service Instances
# =============================================================================

_embedding_service: Optional[EmbeddingService] = None
_vector_store: Optional[VectorStoreService] = None
_llm_service: Optional[LLMService] = None
_retriever_service: Optional[RetrieverService] = None
_ingestion_service: Optional[IngestionService] = None


# =============================================================================
# Dependency Providers
# =============================================================================

def get_embedding_service() -> EmbeddingService:
    """
    Get singleton EmbeddingService instance.
    
    Returns:
        EmbeddingService for generating embeddings.
    """
    global _embedding_service
    
    if _embedding_service is None:
        settings = get_settings()
        _embedding_service = EmbeddingService(settings)
        logger.debug("Created EmbeddingService singleton")
    
    return _embedding_service


def get_vector_store() -> VectorStoreService:
    """
    Get singleton VectorStoreService instance.
    
    Returns:
        VectorStoreService for vector storage and search.
    """
    global _vector_store
    
    if _vector_store is None:
        settings = get_settings()
        _vector_store = VectorStoreService(settings)
        logger.debug("Created VectorStoreService singleton")
    
    return _vector_store


def get_llm_service() -> LLMService:
    """
    Get singleton LLMService instance.
    
    Returns:
        LLMService for answer generation.
    """
    global _llm_service
    
    if _llm_service is None:
        settings = get_settings()
        _llm_service = LLMService(settings)
        logger.debug("Created LLMService singleton")
    
    return _llm_service


def get_retriever_service() -> RetrieverService:
    """
    Get singleton RetrieverService instance.
    
    Composes EmbeddingService, VectorStoreService, and LLMService.
    
    Returns:
        RetrieverService for the complete RAG pipeline.
    """
    global _retriever_service
    
    if _retriever_service is None:
        settings = get_settings()
        _retriever_service = RetrieverService(
            settings=settings,
            embedding_service=get_embedding_service(),
            vector_store=get_vector_store(),
            llm_service=get_llm_service(),
        )
        logger.debug("Created RetrieverService singleton")
    
    return _retriever_service


def get_ingestion_service() -> IngestionService:
    """
    Get singleton IngestionService instance.
    
    Returns:
        IngestionService for document ingestion.
    """
    global _ingestion_service
    
    if _ingestion_service is None:
        settings = get_settings()
        _ingestion_service = IngestionService(settings)
        logger.debug("Created IngestionService singleton")
    
    return _ingestion_service


# =============================================================================
# Reset Functions (for testing)
# =============================================================================

def reset_services() -> None:
    """
    Reset all singleton services.
    
    Useful for testing or when configuration changes.
    """
    global _embedding_service, _vector_store, _llm_service
    global _retriever_service, _ingestion_service
    
    _embedding_service = None
    _vector_store = None
    _llm_service = None
    _retriever_service = None
    _ingestion_service = None
    
    logger.info("Reset all singleton services")
