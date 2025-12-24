"""
Embedding service for generating deterministic document and query embeddings.
Uses OpenAI's text-embedding models for high-quality vector representations.
"""

import hashlib
from typing import Optional
import structlog

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.embeddings import BaseEmbedding

from app.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """
    Service for generating embeddings using OpenAI models.
    Ensures deterministic embeddings for rebuild-safe indexing.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the embedding service.
        
        Args:
            settings: Application settings. If None, loads from environment.
        """
        self._settings = settings or get_settings()
        self._embedding_model: Optional[BaseEmbedding] = None
        self._embedding_cache: dict[str, list[float]] = {}
        
    @property
    def embedding_model(self) -> BaseEmbedding:
        """
        Get or create the embedding model instance.
        Uses lazy initialization for efficiency.
        """
        if self._embedding_model is None:
            self._embedding_model = OpenAIEmbedding(
                model=self._settings.embedding_model,
                api_key=self._settings.openai_api_key,
                dimensions=self._settings.embedding_dimension,
            )
            logger.info(
                "Initialized embedding model",
                model=self._settings.embedding_model,
                dimensions=self._settings.embedding_dimension
            )
        return self._embedding_model
    
    def _compute_text_hash(self, text: str) -> str:
        """
        Compute a deterministic hash for text.
        Used for caching and rebuild-safe indexing.
        
        Args:
            text: The text to hash.
            
        Returns:
            SHA-256 hash of the text.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    async def get_embedding(self, text: str, use_cache: bool = True) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed.
            use_cache: Whether to use cached embeddings if available.
            
        Returns:
            Embedding vector as a list of floats.
        """
        text_hash = self._compute_text_hash(text)
        
        # Check cache first
        if use_cache and text_hash in self._embedding_cache:
            logger.debug("Cache hit for embedding", hash=text_hash[:16])
            return self._embedding_cache[text_hash]
        
        # Generate new embedding
        embedding = await self.embedding_model.aget_text_embedding(text)
        
        # Cache the result
        if use_cache:
            self._embedding_cache[text_hash] = embedding
            
        logger.debug(
            "Generated embedding",
            text_length=len(text),
            embedding_dim=len(embedding)
        )
        
        return embedding
    
    async def get_embeddings_batch(
        self,
        texts: list[str],
        use_cache: bool = True
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed.
            use_cache: Whether to use cached embeddings.
            
        Returns:
            List of embedding vectors.
        """
        embeddings: list[list[float]] = []
        texts_to_embed: list[tuple[int, str]] = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            text_hash = self._compute_text_hash(text)
            if use_cache and text_hash in self._embedding_cache:
                embeddings.append(self._embedding_cache[text_hash])
            else:
                texts_to_embed.append((i, text))
                embeddings.append([])  # Placeholder
        
        # Batch embed uncached texts
        if texts_to_embed:
            uncached_texts = [t[1] for t in texts_to_embed]
            new_embeddings = await self.embedding_model.aget_text_embedding_batch(
                uncached_texts
            )
            
            # Fill in the results and cache
            for (idx, text), embedding in zip(texts_to_embed, new_embeddings):
                embeddings[idx] = embedding
                if use_cache:
                    text_hash = self._compute_text_hash(text)
                    self._embedding_cache[text_hash] = embedding
        
        logger.info(
            "Generated batch embeddings",
            total=len(texts),
            cache_hits=len(texts) - len(texts_to_embed),
            new_embeddings=len(texts_to_embed)
        )
        
        return embeddings
    
    async def get_query_embedding(self, query: str) -> list[float]:
        """
        Generate embedding for a query.
        Query embeddings are not cached as queries are typically unique.
        
        Args:
            query: The query text to embed.
            
        Returns:
            Query embedding vector.
        """
        embedding = await self.embedding_model.aget_query_embedding(query)
        logger.debug("Generated query embedding", query_length=len(query))
        return embedding
    
    def clear_cache(self) -> int:
        """
        Clear the embedding cache.
        
        Returns:
            Number of cached embeddings cleared.
        """
        count = len(self._embedding_cache)
        self._embedding_cache.clear()
        logger.info("Cleared embedding cache", count=count)
        return count
    
    async def health_check(self) -> dict[str, str]:
        """
        Check if the embedding service is healthy.
        
        Returns:
            Health status dictionary.
        """
        try:
            # Try to generate a test embedding
            test_embedding = await self.get_embedding("health check", use_cache=False)
            if len(test_embedding) == self._settings.embedding_dimension:
                return {"status": "healthy", "model": self._settings.embedding_model}
            return {"status": "unhealthy", "error": "Unexpected embedding dimension"}
        except Exception as e:
            logger.error("Embedding health check failed", error=str(e))
            return {"status": "unhealthy", "error": str(e)}
