"""
Document ingestion pipeline.

Orchestrates the complete ingestion flow:
1. Load documents (PDF, DOCX, Markdown)
2. Chunk into semantic segments
3. Store chunks (embeddings generated separately)

Features:
- Idempotent processing (same input = same output)
- Metadata preservation throughout pipeline
- No direct embedding generation (handled by caller)
"""

import hashlib
import time
import uuid
from pathlib import Path
from typing import Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import structlog

from app.config import Settings, get_settings
from app.ingestion.loader import DocumentLoader, Document
from app.ingestion.chunker import SemanticChunker, Chunk
from app.schemas.documents import DocumentType, IngestRequest, IngestResponse, DocumentInfo

logger = structlog.get_logger(__name__)


@dataclass
class ProcessedDocument:
    """Result of processing a single document."""
    document_id: str
    source: str
    document_type: str
    chunks: list[Chunk]
    metadata: dict[str, Any]
    content_hash: str


class IngestionPipeline:
    """
    Orchestrates document ingestion without generating embeddings.
    
    Pipeline flow:
    1. Load document using appropriate loader
    2. Split into semantic chunks with overlap
    3. Return processed chunks with metadata
    
    Embedding generation is NOT done here - that's the caller's responsibility.
    This ensures separation of concerns and flexibility.
    
    Idempotency:
    - Document IDs are generated from content hash
    - Same content = same document ID
    - Re-ingesting same content is safe
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the ingestion pipeline."""
        self._settings = settings or get_settings()
        self._loader = DocumentLoader()
        self._chunker = SemanticChunker(self._settings)
        
        logger.info("Initialized IngestionPipeline")
    
    def _compute_content_hash(self, content: str) -> str:
        """Compute deterministic hash of content for idempotency."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    
    def _generate_document_id(self, content_hash: str) -> str:
        """Generate document ID from content hash."""
        return f"doc_{content_hash[:16]}"
    
    def process_file(
        self,
        file_path: str | Path,
        doc_type: Optional[str] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> ProcessedDocument:
        """
        Process a file through the ingestion pipeline.
        
        Args:
            file_path: Path to the document file.
            doc_type: Document type (auto-detected if None).
            extra_metadata: Additional metadata to attach.
            
        Returns:
            ProcessedDocument with chunks ready for embedding.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect type
        if doc_type is None:
            ext = path.suffix.lower().lstrip(".")
            type_map = {"pdf": "pdf", "docx": "docx", "md": "markdown"}
            doc_type = type_map.get(ext)
            
            if doc_type is None:
                raise ValueError(f"Cannot determine type for: {path.suffix}")
        
        logger.info(
            "Processing file",
            file=path.name,
            type=doc_type,
        )
        
        # Step 1: Load documents
        documents = self._loader.load(path, doc_type=doc_type)
        
        if not documents:
            raise ValueError(f"No content extracted from: {file_path}")
        
        # Combine all document text for content hash
        all_text = "\n\n".join([doc.text for doc in documents])
        content_hash = self._compute_content_hash(all_text)
        document_id = self._generate_document_id(content_hash)
        
        # Step 2: Chunk all documents
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = self._chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        
        # Attach document_id to all chunk metadata
        for chunk in all_chunks:
            chunk.metadata["document_id"] = document_id
        
        # Build result
        result = ProcessedDocument(
            document_id=document_id,
            source=path.name,
            document_type=doc_type,
            chunks=all_chunks,
            metadata={
                "file_path": str(path.absolute()),
                "file_size": path.stat().st_size,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                **(extra_metadata or {}),
            },
            content_hash=content_hash,
        )
        
        logger.info(
            "File processed",
            document_id=document_id,
            source=path.name,
            chunks=len(all_chunks),
        )
        
        return result
    
    def process_markdown_content(
        self,
        content: str,
        source_name: str = "markdown_content",
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> ProcessedDocument:
        """
        Process raw markdown content.
        
        Args:
            content: Raw markdown string.
            source_name: Name to use as source.
            extra_metadata: Additional metadata.
            
        Returns:
            ProcessedDocument with chunks.
        """
        if not content or not content.strip():
            raise ValueError("Empty content provided")
        
        logger.info(
            "Processing markdown content",
            source=source_name,
            length=len(content),
        )
        
        # Compute hash for idempotency
        content_hash = self._compute_content_hash(content)
        document_id = self._generate_document_id(content_hash)
        
        # Load markdown
        documents = self._loader.load_markdown_content(content, source_name)
        
        if not documents:
            raise ValueError("No content extracted from markdown")
        
        # Chunk documents
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = self._chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        
        # Attach document_id
        for chunk in all_chunks:
            chunk.metadata["document_id"] = document_id
        
        result = ProcessedDocument(
            document_id=document_id,
            source=source_name,
            document_type="markdown",
            chunks=all_chunks,
            metadata={
                "content_length": len(content),
                "processed_at": datetime.now(timezone.utc).isoformat(),
                **(extra_metadata or {}),
            },
            content_hash=content_hash,
        )
        
        logger.info(
            "Markdown processed",
            document_id=document_id,
            chunks=len(all_chunks),
        )
        
        return result
    
    def process_request(self, request: IngestRequest) -> ProcessedDocument:
        """
        Process an ingestion request.
        
        Routes to appropriate processing method based on request type.
        
        Args:
            request: IngestRequest from API.
            
        Returns:
            ProcessedDocument ready for embedding.
        """
        doc_type = request.document_type.value
        
        if doc_type in ("pdf", "docx"):
            if not request.file_path:
                raise ValueError(f"{doc_type.upper()} requires file_path")
            
            return self.process_file(
                file_path=request.file_path,
                doc_type=doc_type,
                extra_metadata=request.metadata,
            )
        
        elif doc_type == "markdown":
            if request.content:
                return self.process_markdown_content(
                    content=request.content,
                    source_name=request.metadata.get("source", "markdown_content"),
                    extra_metadata=request.metadata,
                )
            elif request.file_path:
                return self.process_file(
                    file_path=request.file_path,
                    doc_type="markdown",
                    extra_metadata=request.metadata,
                )
            else:
                raise ValueError("Markdown requires content or file_path")
        
        else:
            raise ValueError(f"Unsupported document type: {doc_type}")
    
    def chunks_to_dicts(self, chunks: list[Chunk]) -> list[dict[str, Any]]:
        """
        Convert Chunk objects to dictionaries for vector store.
        
        Args:
            chunks: List of Chunk objects.
            
        Returns:
            List of dictionaries with text and metadata.
        """
        return [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source,
                "page_number": chunk.page_number,
                "section": chunk.section,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]


class IngestionService:
    """
    High-level ingestion service that connects pipeline to vector store.
    
    Coordinates:
    1. Document processing (pipeline)
    2. Embedding generation (embedding service)
    3. Vector storage (vector store)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        pipeline: Optional[IngestionPipeline] = None,
    ):
        """Initialize ingestion service."""
        self._settings = settings or get_settings()
        self._pipeline = pipeline or IngestionPipeline(self._settings)
    
    @property
    def pipeline(self) -> IngestionPipeline:
        """Get the ingestion pipeline."""
        return self._pipeline
    
    async def ingest(
        self,
        request: IngestRequest,
        embedding_service,  # EmbeddingService
        vector_store,  # VectorStoreService
    ) -> IngestResponse:
        """
        Complete ingestion: process, embed, and store.
        
        Args:
            request: Ingestion request.
            embedding_service: Service for generating embeddings.
            vector_store: Service for storing vectors.
            
        Returns:
            IngestResponse with results.
        """
        start_time = time.perf_counter()
        
        try:
            # Step 1: Process document
            processed = self._pipeline.process_request(request)
            
            if not processed.chunks:
                return IngestResponse(
                    success=False,
                    message="No content extracted from document",
                    documents=[],
                    total_chunks=0,
                    processing_time_ms=(time.perf_counter() - start_time) * 1000,
                )
            
            # Step 2: Generate embeddings for all chunks
            chunk_texts = [chunk.text for chunk in processed.chunks]
            embeddings = await embedding_service.get_embeddings_batch(chunk_texts)
            
            # Step 3: Store in vector store
            chunk_dicts = self._pipeline.chunks_to_dicts(processed.chunks)
            
            await vector_store.add_chunks(
                chunks=chunk_dicts,
                embeddings=embeddings,
                document_id=processed.document_id,
                document_metadata={
                    "source": processed.source,
                    "document_type": processed.document_type,
                    **processed.metadata,
                },
            )
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Build response
            doc_info = DocumentInfo(
                document_id=processed.document_id,
                source=processed.source,
                document_type=DocumentType(processed.document_type),
                chunk_count=len(processed.chunks),
                ingested_at=datetime.now(timezone.utc),
                metadata=processed.metadata,
            )
            
            logger.info(
                "Document ingested",
                document_id=processed.document_id,
                chunks=len(processed.chunks),
                time_ms=round(processing_time, 2),
            )
            
            return IngestResponse(
                success=True,
                message=f"Successfully ingested with {len(processed.chunks)} chunks",
                documents=[doc_info],
                total_chunks=len(processed.chunks),
                processing_time_ms=round(processing_time, 2),
            )
            
        except FileNotFoundError as e:
            logger.error("File not found", error=str(e))
            return IngestResponse(
                success=False,
                message=str(e),
                documents=[],
                total_chunks=0,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        except Exception as e:
            logger.error("Ingestion failed", error=str(e), exc_info=True)
            return IngestResponse(
                success=False,
                message=f"Ingestion failed: {str(e)}",
                documents=[],
                total_chunks=0,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
