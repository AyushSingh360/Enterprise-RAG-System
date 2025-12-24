"""
Configuration management for the Enterprise RAG system.
Uses Pydantic Settings for type-safe configuration with environment variable support.
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    All settings have sensible defaults for development.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application Settings
    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    # OpenAI Configuration
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for embeddings and LLM"
    )
    
    # Vector Store Settings
    faiss_index_path: Path = Field(
        default=Path("./data/faiss_index"),
        description="Path to store FAISS index"
    )
    document_store_path: Path = Field(
        default=Path("./data/documents"),
        description="Path to store ingested documents"
    )
    
    # Embedding Settings
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model to use"
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Dimension of embedding vectors"
    )
    
    # LLM Settings
    llm_model: str = Field(
        default="gpt-4-turbo-preview",
        description="OpenAI LLM model for generation"
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature (0.0 for deterministic outputs)"
    )
    llm_max_tokens: int = Field(
        default=1024,
        ge=1,
        le=4096,
        description="Maximum tokens in LLM response"
    )
    
    # Retrieval Settings
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for retrieval"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of documents to retrieve"
    )
    
    # Chunking Settings
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=2048,
        description="Size of document chunks in tokens"
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between consecutive chunks"
    )
    
    @field_validator("faiss_index_path", "document_store_path", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v
    
    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.faiss_index_path.mkdir(parents=True, exist_ok=True)
        self.document_store_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == "production"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure settings are loaded only once.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings
