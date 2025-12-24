"""
Vector store service using FAISS for efficient similarity search.
Handles index persistence, document storage, and retrieval operations.
"""

import json
import uuid
from pathlib import Path
from typing import Optional, Any
from datetime import datetime
import structlog

import faiss
import numpy as np
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

from app.config import Settings, get_settings
from app.core.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)


class VectorStoreService:
    """
    Service for managing the FAISS vector store.
    Handles document indexing, persistence, and similarity search.
    """
    
    DOCUMENT_METADATA_FILE = "document_metadata.json"
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        """
        Initialize the vector store service.
        
        Args:
            settings: Application settings.
            embedding_service: Embedding service for generating vectors.
        """
        self._settings = settings or get_settings()
        self._embedding_service = embedding_service or EmbeddingService(self._settings)
        
        self._faiss_index: Optional[faiss.Index] = None
        self._vector_store: Optional[FaissVectorStore] = None
        self._index: Optional[VectorStoreIndex] = None
        self._document_metadata: dict[str, dict[str, Any]] = {}
        
        self._initialize_store()
    
    def _initialize_store(self) -> None:
        """Initialize or load the FAISS index and document metadata."""
        index_path = self._settings.faiss_index_path
        
        if self._index_exists():
            self._load_index()
        else:
            self._create_new_index()
        
        self._load_document_metadata()
    
    def _index_exists(self) -> bool:
        """Check if a persisted index exists."""
        index_file = self._settings.faiss_index_path / "default__vector_store.faiss"
        return index_file.exists()
    
    def _create_new_index(self) -> None:
        """Create a new FAISS index."""
        # Create a flat L2 index for exact search
        # In production, consider using IVF for larger datasets
        d = self._settings.embedding_dimension
        self._faiss_index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
        
        self._vector_store = FaissVectorStore(faiss_index=self._faiss_index)
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )
        
        # Create empty index
        self._index = VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=self._embedding_service.embedding_model,
        )
        
        logger.info(
            "Created new FAISS index",
            dimension=d,
            index_type="FlatIP"
        )
    
    def _load_index(self) -> None:
        """Load an existing FAISS index from disk."""
        index_path = self._settings.faiss_index_path
        
        # Load the FAISS index
        self._vector_store = FaissVectorStore.from_persist_dir(str(index_path))
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store,
            persist_dir=str(index_path)
        )
        
        # Load the index
        self._index = VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=self._embedding_service.embedding_model,
        )
        
        logger.info("Loaded existing FAISS index", path=str(index_path))
    
    def _load_document_metadata(self) -> None:
        """Load document metadata from disk."""
        metadata_path = self._settings.document_store_path / self.DOCUMENT_METADATA_FILE
        
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self._document_metadata = json.load(f)
            logger.info(
                "Loaded document metadata",
                document_count=len(self._document_metadata)
            )
        else:
            self._document_metadata = {}
            logger.info("No existing document metadata found")
    
    def _save_document_metadata(self) -> None:
        """Save document metadata to disk."""
        metadata_path = self._settings.document_store_path / self.DOCUMENT_METADATA_FILE
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._document_metadata, f, indent=2, default=str)
        
        logger.debug("Saved document metadata")
    
    def persist(self) -> None:
        """Persist the index and metadata to disk."""
        index_path = self._settings.faiss_index_path
        
        if self._index is not None:
            self._index.storage_context.persist(persist_dir=str(index_path))
        
        self._save_document_metadata()
        
        logger.info("Persisted vector store", path=str(index_path))
    
    async def add_nodes(
        self,
        nodes: list[TextNode],
        document_id: str,
        document_metadata: dict[str, Any]
    ) -> int:
        """
        Add text nodes to the vector store.
        
        Args:
            nodes: List of TextNode objects to add.
            document_id: ID of the parent document.
            document_metadata: Metadata for the parent document.
            
        Returns:
            Number of nodes added.
        """
        if not nodes:
            return 0
        
        # Ensure all nodes have proper metadata
        for node in nodes:
            node.metadata["document_id"] = document_id
            if "node_id" not in node.metadata:
                node.metadata["node_id"] = str(uuid.uuid4())
        
        # Add nodes to index
        self._index.insert_nodes(nodes)
        
        # Store document metadata
        self._document_metadata[document_id] = {
            **document_metadata,
            "chunk_count": len(nodes),
            "indexed_at": datetime.utcnow().isoformat()
        }
        
        # Persist changes
        self.persist()
        
        logger.info(
            "Added nodes to vector store",
            document_id=document_id,
            node_count=len(nodes)
        )
        
        return len(nodes)
    
    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        metadata_filter: Optional[dict[str, Any]] = None
    ) -> list[NodeWithScore]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query_embedding: Query vector.
            top_k: Maximum number of results.
            similarity_threshold: Minimum similarity score.
            metadata_filter: Filter by metadata fields.
            
        Returns:
            List of NodeWithScore objects sorted by similarity.
        """
        k = top_k or self._settings.top_k
        threshold = similarity_threshold or self._settings.similarity_threshold
        
        # Create retriever
        retriever = self._index.as_retriever(
            similarity_top_k=k,
        )
        
        # Perform query
        from llama_index.core.vector_stores import VectorStoreQuery
        
        query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=k,
        )
        
        results = self._vector_store.query(query)
        
        # Convert to NodeWithScore objects
        nodes_with_scores: list[NodeWithScore] = []
        
        if results.nodes and results.similarities:
            for node, score in zip(results.nodes, results.similarities):
                # Apply threshold filter
                if score >= threshold:
                    # Apply metadata filter if provided
                    if metadata_filter:
                        if not self._matches_filter(node.metadata, metadata_filter):
                            continue
                    
                    nodes_with_scores.append(
                        NodeWithScore(node=node, score=score)
                    )
        
        # Sort by score descending
        nodes_with_scores.sort(key=lambda x: x.score, reverse=True)
        
        # Limit to top_k
        nodes_with_scores = nodes_with_scores[:k]
        
        logger.info(
            "Similarity search completed",
            results_count=len(nodes_with_scores),
            threshold=threshold,
            top_k=k
        )
        
        return nodes_with_scores
    
    def _matches_filter(
        self,
        metadata: dict[str, Any],
        filter_dict: dict[str, Any]
    ) -> bool:
        """Check if metadata matches the filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
    
    def get_document_info(self, document_id: str) -> Optional[dict[str, Any]]:
        """Get metadata for a specific document."""
        return self._document_metadata.get(document_id)
    
    def get_all_documents(self) -> dict[str, dict[str, Any]]:
        """Get metadata for all documents."""
        return self._document_metadata.copy()
    
    def get_index_stats(self) -> dict[str, Any]:
        """Get statistics about the vector index."""
        return {
            "total_documents": len(self._document_metadata),
            "total_vectors": self._faiss_index.ntotal if self._faiss_index else 0,
            "dimension": self._settings.embedding_dimension,
            "index_path": str(self._settings.faiss_index_path)
        }
    
    async def health_check(self) -> dict[str, str]:
        """Check if the vector store is healthy."""
        try:
            stats = self.get_index_stats()
            return {
                "status": "healthy",
                "total_documents": str(stats["total_documents"]),
                "total_vectors": str(stats["total_vectors"])
            }
        except Exception as e:
            logger.error("Vector store health check failed", error=str(e))
            return {"status": "unhealthy", "error": str(e)}
