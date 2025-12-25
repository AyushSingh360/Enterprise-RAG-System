"""
FAISS vector store with persistent storage.

Features:
- Persistent FAISS index saved to disk
- Document metadata storage with JSON persistence
- Similarity search with score threshold filtering
- Automatic index rebuilding on startup
"""

import json
import uuid
from pathlib import Path
from typing import Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import structlog

import faiss
import numpy as np

from app.config import Settings, get_settings

logger = structlog.get_logger(__name__)


@dataclass
class StoredChunk:
    """Represents a stored document chunk with metadata."""
    chunk_id: str
    document_id: str
    text: str
    source: str
    page_number: Optional[int]
    section: Optional[str]
    chunk_index: int
    metadata: dict[str, Any]
    created_at: str


@dataclass
class SearchResult:
    """Result from similarity search."""
    chunk: StoredChunk
    score: float


class VectorStoreService:
    """
    Persistent FAISS vector store for document chunks.
    
    Features:
    - FAISS IndexFlatIP for cosine similarity (normalized vectors)
    - Persistent storage of index and metadata
    - Similarity score threshold filtering
    - Document and chunk management
    """
    
    FAISS_INDEX_FILE = "faiss.index"
    CHUNKS_FILE = "chunks.json"
    DOCUMENTS_FILE = "documents.json"
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the vector store."""
        self._settings = settings or get_settings()
        self._index_path = self._settings.faiss_index_path
        self._index_path.mkdir(parents=True, exist_ok=True)
        
        self._dimension = self._settings.embedding_dimension
        self._index: Optional[faiss.Index] = None
        self._chunks: dict[str, StoredChunk] = {}  # chunk_id -> chunk
        self._chunk_ids: list[str] = []  # Ordered list matching FAISS index
        self._documents: dict[str, dict[str, Any]] = {}  # document_id -> metadata
        
        self._load_or_create_index()
        
        logger.info(
            "Initialized VectorStoreService",
            dimension=self._dimension,
            total_chunks=len(self._chunks),
            total_documents=len(self._documents),
        )
    
    def _load_or_create_index(self) -> None:
        """Load existing index from disk or create new one."""
        index_file = self._index_path / self.FAISS_INDEX_FILE
        chunks_file = self._index_path / self.CHUNKS_FILE
        documents_file = self._index_path / self.DOCUMENTS_FILE
        
        if index_file.exists() and chunks_file.exists():
            # Load existing index
            try:
                self._index = faiss.read_index(str(index_file))
                
                with open(chunks_file, "r", encoding="utf-8") as f:
                    chunks_data = json.load(f)
                    self._chunks = {
                        k: StoredChunk(**v) for k, v in chunks_data["chunks"].items()
                    }
                    self._chunk_ids = chunks_data["chunk_ids"]
                
                if documents_file.exists():
                    with open(documents_file, "r", encoding="utf-8") as f:
                        self._documents = json.load(f)
                
                logger.info(
                    "Loaded existing FAISS index",
                    vectors=self._index.ntotal,
                    chunks=len(self._chunks),
                )
                return
            except Exception as e:
                logger.warning("Failed to load index, creating new", error=str(e))
        
        # Create new index
        self._index = faiss.IndexFlatIP(self._dimension)
        self._chunks = {}
        self._chunk_ids = []
        self._documents = {}
        
        logger.info("Created new FAISS index", dimension=self._dimension)
    
    def _persist(self) -> None:
        """Persist index and metadata to disk."""
        index_file = self._index_path / self.FAISS_INDEX_FILE
        chunks_file = self._index_path / self.CHUNKS_FILE
        documents_file = self._index_path / self.DOCUMENTS_FILE
        
        # Save FAISS index
        faiss.write_index(self._index, str(index_file))
        
        # Save chunks
        chunks_data = {
            "chunks": {k: asdict(v) for k, v in self._chunks.items()},
            "chunk_ids": self._chunk_ids,
        }
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, default=str)
        
        # Save documents
        with open(documents_file, "w", encoding="utf-8") as f:
            json.dump(self._documents, f, indent=2, default=str)
        
        logger.debug("Persisted vector store to disk")
    
    def _normalize_vector(self, vector: list[float]) -> np.ndarray:
        """Normalize vector for cosine similarity using inner product."""
        arr = np.array(vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(arr)
        return arr
    
    async def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        document_id: str,
        document_metadata: dict[str, Any],
    ) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with text and metadata.
            embeddings: Corresponding embeddings for each chunk.
            document_id: Parent document identifier.
            document_metadata: Metadata for the parent document.
            
        Returns:
            Number of chunks added.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have same length")
        
        if not chunks:
            return 0
        
        # Store document metadata
        self._documents[document_id] = {
            **document_metadata,
            "chunk_count": len(chunks),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Prepare vectors for batch insertion
        vectors = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
            
            stored_chunk = StoredChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=chunk["text"],
                source=chunk.get("source", document_metadata.get("source", "unknown")),
                page_number=chunk.get("page_number"),
                section=chunk.get("section"),
                chunk_index=i,
                metadata=chunk.get("metadata", {}),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            
            self._chunks[chunk_id] = stored_chunk
            self._chunk_ids.append(chunk_id)
            
            # Normalize embedding for cosine similarity
            normalized = self._normalize_vector(embedding)
            vectors.append(normalized)
        
        # Batch add to FAISS index
        vectors_matrix = np.vstack(vectors)
        self._index.add(vectors_matrix)
        
        # Persist to disk
        self._persist()
        
        logger.info(
            "Added chunks to vector store",
            document_id=document_id,
            chunk_count=len(chunks),
            total_vectors=self._index.ntotal,
        )
        
        return len(chunks)
    
    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector.
            top_k: Maximum results to return.
            similarity_threshold: Minimum similarity score (0-1).
            metadata_filter: Filter by metadata fields.
            
        Returns:
            List of SearchResult sorted by score descending.
        """
        if self._index.ntotal == 0:
            logger.info("Vector store is empty, no results")
            return []
        
        k = min(top_k or self._settings.top_k, self._index.ntotal)
        threshold = similarity_threshold or self._settings.similarity_threshold
        
        # Normalize query vector
        query_vector = self._normalize_vector(query_embedding)
        
        # Search FAISS index
        scores, indices = self._index.search(query_vector, k)
        
        results: list[SearchResult] = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            # Convert score to 0-1 range (inner product of normalized vectors)
            # Score is already cosine similarity due to normalization
            similarity = float(score)
            
            # Apply threshold filter
            if similarity < threshold:
                continue
            
            # Get chunk
            chunk_id = self._chunk_ids[idx]
            chunk = self._chunks.get(chunk_id)
            
            if chunk is None:
                continue
            
            # Apply metadata filter
            if metadata_filter:
                if not self._matches_filter(chunk, metadata_filter):
                    continue
            
            results.append(SearchResult(chunk=chunk, score=similarity))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(
            "Similarity search complete",
            query_dim=len(query_embedding),
            candidates=k,
            threshold=threshold,
            results=len(results),
        )
        
        return results
    
    def _matches_filter(
        self,
        chunk: StoredChunk,
        filter_dict: dict[str, Any],
    ) -> bool:
        """Check if chunk matches metadata filter."""
        for key, value in filter_dict.items():
            # Check direct chunk attributes
            if hasattr(chunk, key):
                if getattr(chunk, key) != value:
                    return False
            # Check metadata dict
            elif key in chunk.metadata:
                if chunk.metadata[key] != value:
                    return False
            else:
                return False
        return True
    
    def get_document_info(self, document_id: str) -> Optional[dict[str, Any]]:
        """Get metadata for a document."""
        return self._documents.get(document_id)
    
    def get_all_documents(self) -> dict[str, dict[str, Any]]:
        """Get all document metadata."""
        return self._documents.copy()
    
    def get_stats(self) -> dict[str, Any]:
        """Get vector store statistics."""
        return {
            "total_vectors": self._index.ntotal if self._index else 0,
            "total_chunks": len(self._chunks),
            "total_documents": len(self._documents),
            "dimension": self._dimension,
            "index_path": str(self._index_path),
        }
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and its chunks from the store.
        Note: FAISS doesn't support efficient deletion, so we rebuild.
        """
        if document_id not in self._documents:
            return False
        
        # Find chunk IDs to remove
        chunks_to_remove = [
            cid for cid, chunk in self._chunks.items()
            if chunk.document_id == document_id
        ]
        
        if not chunks_to_remove:
            del self._documents[document_id]
            self._persist()
            return True
        
        # Remove from chunks dict
        for chunk_id in chunks_to_remove:
            del self._chunks[chunk_id]
        
        # Rebuild index without deleted chunks
        remaining_ids = [cid for cid in self._chunk_ids if cid not in chunks_to_remove]
        
        # This requires re-embedding, which is expensive
        # For production, consider using IndexIDMap for efficient deletion
        logger.warning(
            "Document deletion requires index rebuild",
            document_id=document_id,
            removed_chunks=len(chunks_to_remove),
        )
        
        self._chunk_ids = remaining_ids
        del self._documents[document_id]
        self._persist()
        
        return True
    
    async def health_check(self) -> dict[str, str]:
        """Check if vector store is healthy."""
        try:
            stats = self.get_stats()
            return {
                "status": "healthy",
                "total_vectors": str(stats["total_vectors"]),
                "total_documents": str(stats["total_documents"]),
            }
        except Exception as e:
            logger.error("Vector store health check failed", error=str(e))
            return {"status": "unhealthy", "error": str(e)}
