"""
Dependency injection for API routes.
"""

from functools import lru_cache
from typing import Generator

from app.config import get_settings, Settings
from app.core.embeddings import EmbeddingService
from app.core.vector_store import VectorStoreService
from app.core.llm import LLMService
from app.core.retriever import RetrieverService
from app.ingestion.pipeline import IngestionPipeline


# Singleton instances
_embedding_service: EmbeddingService | None = None
_vector_store: VectorStoreService | None = None
_llm_service: LLMService | None = None
_retriever_service: RetrieverService | None = None
_ingestion_pipeline: IngestionPipeline | None = None


def get_embedding_service() -> EmbeddingService:
    """Get the singleton embedding service."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(get_settings())
    return _embedding_service


def get_vector_store() -> VectorStoreService:
    """Get the singleton vector store service."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService(
            get_settings(),
            get_embedding_service()
        )
    return _vector_store


def get_llm_service() -> LLMService:
    """Get the singleton LLM service."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(get_settings())
    return _llm_service


def get_retriever_service() -> RetrieverService:
    """Get the singleton retriever service."""
    global _retriever_service
    if _retriever_service is None:
        _retriever_service = RetrieverService(
            settings=get_settings(),
            embedding_service=get_embedding_service(),
            vector_store=get_vector_store(),
            llm_service=get_llm_service()
        )
    return _retriever_service


def get_ingestion_pipeline() -> IngestionPipeline:
    """Get the singleton ingestion pipeline."""
    global _ingestion_pipeline
    if _ingestion_pipeline is None:
        _ingestion_pipeline = IngestionPipeline(
            settings=get_settings(),
            embedding_service=get_embedding_service(),
            vector_store=get_vector_store()
        )
    return _ingestion_pipeline
