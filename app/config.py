"""
Configuration management for the Enterprise RAG system.
Uses Pydantic BaseSettings for type-safe, environment-based configuration.

All secrets are loaded from environment variables or .env file.
No hardcoded secrets are present in this module.
"""

import os
import sys
from pathlib import Path
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Priority order:
    1. Environment variables
    2. .env file
    3. Default values
    
    Secrets use SecretStr to prevent accidental logging.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="",  # No prefix required
    )
    
    # ===================
    # Application Settings
    # ===================
    app_name: str = Field(
        default="Enterprise RAG System",
        description="Application name for logging and identification"
    )
    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (do not use in production)"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging verbosity level"
    )
    log_format: Literal["json", "console"] = Field(
        default="json",
        description="Log output format"
    )
    
    # ===================
    # API Settings
    # ===================
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="API server port"
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS origins"
    )
    
    # ===================
    # OpenAI Configuration (Secrets)
    # ===================
    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API key (required for embeddings and LLM)"
    )
    openai_org_id: Optional[str] = Field(
        default=None,
        description="Optional OpenAI organization ID"
    )
    
    # ===================
    # Storage Paths
    # ===================
    faiss_index_path: Path = Field(
        default=Path("./data/faiss_index"),
        description="Directory path for FAISS index persistence"
    )
    document_store_path: Path = Field(
        default=Path("./data/documents"),
        description="Directory path for document metadata storage"
    )
    
    # ===================
    # Embedding Configuration
    # ===================
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name"
    )
    embedding_dimension: int = Field(
        default=1536,
        ge=1,
        description="Embedding vector dimension"
    )
    embedding_batch_size: int = Field(
        default=100,
        ge=1,
        le=2048,
        description="Batch size for embedding generation"
    )
    
    # ===================
    # LLM Configuration
    # ===================
    llm_model: str = Field(
        default="gpt-4-turbo-preview",
        description="OpenAI LLM model for answer generation"
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
    llm_timeout: int = Field(
        default=60,
        ge=1,
        description="LLM request timeout in seconds"
    )
    
    # ===================
    # Retrieval Configuration
    # ===================
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score for retrieval"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of documents to retrieve"
    )
    
    # ===================
    # Chunking Configuration
    # ===================
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=4096,
        description="Target size of document chunks in tokens"
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between consecutive chunks in tokens"
    )
    
    # ===================
    # Validators
    # ===================
    @field_validator("faiss_index_path", "document_store_path", mode="before")
    @classmethod
    def convert_to_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    # ===================
    # Computed Properties
    # ===================
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_env == "development"
    
    @property
    def openai_api_key_value(self) -> str:
        """Get the actual API key value (use carefully)."""
        return self.openai_api_key.get_secret_value()
    
    # ===================
    # Utility Methods
    # ===================
    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.faiss_index_path.mkdir(parents=True, exist_ok=True)
        self.document_store_path.mkdir(parents=True, exist_ok=True)
    
    def validate_required_secrets(self) -> None:
        """
        Validate that required secrets are configured.
        Raises ValueError if critical configuration is missing.
        """
        if not self.openai_api_key.get_secret_value():
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Set it in your .env file or environment."
            )
    
    def get_safe_dict(self) -> dict:
        """
        Get settings as dictionary with secrets masked.
        Safe for logging.
        """
        data = self.model_dump()
        # Mask sensitive fields
        if "openai_api_key" in data:
            data["openai_api_key"] = "***MASKED***"
        return data


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are loaded only once
    and reused across the application.
    
    Returns:
        Settings: The application settings instance.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings
