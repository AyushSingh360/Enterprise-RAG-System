"""
Embedding service for generating and caching document/query embeddings.

Features:
- Deterministic embeddings with SHA-256 hash-based caching
- Batch processing for efficiency
- Rebuild-safe indexing (same text = same embedding)
- OpenAI text-embedding-3-small model
"""

import hashlib
import json
from pathlib import Path
from typing import Optional
import structlog

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIError

from app.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class EmbeddingCache:
    """
    Persistent disk-based cache for embeddings.
    Uses SHA-256 hash of text as key for deterministic lookups.
    """
    
    def __init__(self, cache_dir: Path):
        """Initialize the embedding cache."""
        self.cache_dir = cache_dir / "embedding_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, list[float]] = {}
        self._load_cache_index()
    
    def _get_hash(self, text: str) -> str:
        """Generate SHA-256 hash for text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def _get_cache_path(self, text_hash: str) -> Path:
        """Get file path for cached embedding."""
        # Use first 2 chars as subdirectory for better filesystem performance
        subdir = self.cache_dir / text_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{text_hash}.json"
    
    def _load_cache_index(self) -> None:
        """Load cache index into memory for fast lookups."""
        index_file = self.cache_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    self._index = set(json.load(f))
            except Exception:
                self._index = set()
        else:
            self._index = set()
    
    def _save_cache_index(self) -> None:
        """Persist cache index to disk."""
        index_file = self.cache_dir / "index.json"
        with open(index_file, "w") as f:
            json.dump(list(self._index), f)
    
    def get(self, text: str) -> Optional[list[float]]:
        """Get cached embedding for text."""
        text_hash = self._get_hash(text)
        
        # Check memory cache first
        if text_hash in self._memory_cache:
            return self._memory_cache[text_hash]
        
        # Check disk cache
        if text_hash in self._index:
            cache_path = self._get_cache_path(text_hash)
            if cache_path.exists():
                try:
                    with open(cache_path, "r") as f:
                        embedding = json.load(f)
                    self._memory_cache[text_hash] = embedding
                    return embedding
                except Exception as e:
                    logger.warning("Cache read failed", hash=text_hash[:16], error=str(e))
        
        return None
    
    def set(self, text: str, embedding: list[float]) -> None:
        """Cache embedding for text."""
        text_hash = self._get_hash(text)
        
        # Store in memory
        self._memory_cache[text_hash] = embedding
        
        # Store to disk
        cache_path = self._get_cache_path(text_hash)
        try:
            with open(cache_path, "w") as f:
                json.dump(embedding, f)
            self._index.add(text_hash)
            self._save_cache_index()
        except Exception as e:
            logger.warning("Cache write failed", hash=text_hash[:16], error=str(e))
    
    def clear(self) -> int:
        """Clear all cached embeddings."""
        count = len(self._index)
        self._memory_cache.clear()
        self._index.clear()
        self._save_cache_index()
        
        # Remove cache files
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                for file in subdir.glob("*.json"):
                    file.unlink()
        
        return count


class EmbeddingService:
    """
    Service for generating embeddings using OpenAI models.
    
    Features:
    - Persistent caching for rebuild-safe indexing
    - Batch processing for efficiency
    - Automatic retry with exponential backoff
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the embedding service."""
        self._settings = settings or get_settings()
        self._client: Optional[AsyncOpenAI] = None
        self._cache = EmbeddingCache(self._settings.document_store_path)
        
        logger.info(
            "Initialized EmbeddingService",
            model=self._settings.embedding_model,
            dimension=self._settings.embedding_dimension,
        )
    
    @property
    def client(self) -> AsyncOpenAI:
        """Get or create AsyncOpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._settings.openai_api_key_value,
                organization=self._settings.openai_org_id,
            )
        return self._client
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=retry_if_exception_type((RateLimitError, APIError)),
    )
    async def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text using OpenAI API."""
        response = await self.client.embeddings.create(
            model=self._settings.embedding_model,
            input=text,
            dimensions=self._settings.embedding_dimension,
        )
        return response.data[0].embedding
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=retry_if_exception_type((RateLimitError, APIError)),
    )
    async def _generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single API call."""
        response = await self.client.embeddings.create(
            model=self._settings.embedding_model,
            input=texts,
            dimensions=self._settings.embedding_dimension,
        )
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
    
    async def get_embedding(self, text: str, use_cache: bool = True) -> list[float]:
        """
        Get embedding for text, using cache if available.
        
        Args:
            text: Text to embed.
            use_cache: Whether to use cached embedding if available.
            
        Returns:
            Embedding vector as list of floats.
        """
        # Check cache
        if use_cache:
            cached = self._cache.get(text)
            if cached is not None:
                logger.debug("Cache hit for embedding", text_length=len(text))
                return cached
        
        # Generate new embedding
        embedding = await self._generate_embedding(text)
        
        # Cache result
        if use_cache:
            self._cache.set(text, embedding)
        
        logger.debug(
            "Generated embedding",
            text_length=len(text),
            dimension=len(embedding),
        )
        
        return embedding
    
    async def get_embeddings_batch(
        self,
        texts: list[str],
        use_cache: bool = True,
    ) -> list[list[float]]:
        """
        Get embeddings for multiple texts with caching.
        
        Args:
            texts: List of texts to embed.
            use_cache: Whether to use cached embeddings.
            
        Returns:
            List of embedding vectors.
        """
        results: list[Optional[list[float]]] = [None] * len(texts)
        texts_to_embed: list[tuple[int, str]] = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            if use_cache:
                cached = self._cache.get(text)
                if cached is not None:
                    results[i] = cached
                    continue
            texts_to_embed.append((i, text))
        
        cache_hits = len(texts) - len(texts_to_embed)
        
        # Generate embeddings for uncached texts in batches
        if texts_to_embed:
            batch_size = self._settings.embedding_batch_size
            
            for batch_start in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[batch_start:batch_start + batch_size]
                batch_texts = [t[1] for t in batch]
                
                embeddings = await self._generate_embeddings_batch(batch_texts)
                
                for (idx, text), embedding in zip(batch, embeddings):
                    results[idx] = embedding
                    if use_cache:
                        self._cache.set(text, embedding)
        
        logger.info(
            "Batch embeddings complete",
            total=len(texts),
            cache_hits=cache_hits,
            generated=len(texts_to_embed),
        )
        
        return results  # type: ignore
    
    async def get_query_embedding(self, query: str) -> list[float]:
        """
        Get embedding for a query (not cached by default).
        
        Args:
            query: Query text to embed.
            
        Returns:
            Query embedding vector.
        """
        # Queries are typically unique, so don't cache by default
        return await self.get_embedding(query, use_cache=False)
    
    def clear_cache(self) -> int:
        """Clear the embedding cache."""
        count = self._cache.clear()
        logger.info("Cleared embedding cache", count=count)
        return count
    
    async def health_check(self) -> dict[str, str]:
        """Check if the embedding service is healthy."""
        try:
            test_embedding = await self.get_embedding("health check test", use_cache=False)
            if len(test_embedding) == self._settings.embedding_dimension:
                return {
                    "status": "healthy",
                    "model": self._settings.embedding_model,
                    "dimension": str(self._settings.embedding_dimension),
                }
            return {"status": "unhealthy", "error": "Unexpected dimension"}
        except Exception as e:
            logger.error("Embedding health check failed", error=str(e))
            return {"status": "unhealthy", "error": str(e)}
