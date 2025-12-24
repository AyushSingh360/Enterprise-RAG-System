"""
Complete ingestion pipeline orchestrating loading, chunking, and indexing.
"""

import time
import uuid
from typing import Optional, Any
from datetime import datetime
import structlog

from app.config import Settings, get_settings
from app.schemas.documents import (
    DocumentType,
    IngestRequest,
    IngestResponse,
    DocumentInfo,
)
from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import SemanticChunker
from app.core.vector_store import VectorStoreService
from app.core.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)


class IngestionPipeline:
    """
    Complete document ingestion pipeline that:
    1. Loads documents from various sources
    2. Chunks them semantically
    3. Generates embeddings
    4. Stores in vector database
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStoreService] = None
    ):
        """Initialize the ingestion pipeline."""
        self._settings = settings or get_settings()
        self._embedding_service = embedding_service or EmbeddingService(self._settings)
        self._vector_store = vector_store or VectorStoreService(
            self._settings, self._embedding_service
        )
        self._loader = DocumentLoader()
        self._chunker = SemanticChunker(self._settings)
    
    async def ingest(self, request: IngestRequest) -> IngestResponse:
        """
        Process a document ingestion request.
        
        Args:
            request: The ingestion request.
            
        Returns:
            IngestResponse with ingestion results.
        """
        start_time = time.perf_counter()
        document_id = f"doc_{uuid.uuid4().hex[:12]}"
        
        try:
            # Step 1: Determine source
            if request.document_type == DocumentType.SQL:
                source = request.sql_query or ""
                kwargs = {"connection_string": request.connection_string}
            elif request.document_type in (DocumentType.PDF, DocumentType.DOCX):
                source = request.file_path or ""
                kwargs = {}
            else:
                source = request.content or ""
                kwargs = {}
            
            # Step 2: Load documents
            documents = await self._loader.load_document(
                doc_type=request.document_type,
                source=source,
                metadata=request.metadata,
                **kwargs
            )
            
            if not documents:
                return IngestResponse(
                    success=False,
                    message="No content extracted from document",
                    documents=[],
                    total_chunks=0,
                    processing_time_ms=(time.perf_counter() - start_time) * 1000
                )
            
            # Step 3: Chunk documents
            nodes = self._chunker.chunk_documents(documents)
            
            # Step 4: Add to vector store
            doc_metadata = {
                "source": documents[0].metadata.get("source", "unknown"),
                "document_type": request.document_type.value,
                **request.metadata
            }
            
            chunk_count = await self._vector_store.add_nodes(
                nodes=nodes,
                document_id=document_id,
                document_metadata=doc_metadata
            )
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            doc_info = DocumentInfo(
                document_id=document_id,
                source=doc_metadata["source"],
                document_type=request.document_type,
                chunk_count=chunk_count,
                ingested_at=datetime.utcnow(),
                metadata=request.metadata
            )
            
            logger.info(
                "Document ingested successfully",
                document_id=document_id,
                chunks=chunk_count,
                time_ms=processing_time
            )
            
            return IngestResponse(
                success=True,
                message=f"Successfully ingested document with {chunk_count} chunks",
                documents=[doc_info],
                total_chunks=chunk_count,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error("Ingestion failed", error=str(e))
            return IngestResponse(
                success=False,
                message=f"Ingestion failed: {str(e)}",
                documents=[],
                total_chunks=0,
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    @property
    def vector_store(self) -> VectorStoreService:
        """Get the vector store service."""
        return self._vector_store
