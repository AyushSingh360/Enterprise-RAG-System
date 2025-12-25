"""
Semantic text chunker with overlap handling.

Features:
- Sentence-aware chunking (doesn't split mid-sentence)
- Configurable chunk size and overlap
- Preserves document metadata through to chunks
- Deterministic chunking for idempotent processing
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Any
import structlog

from app.config import Settings, get_settings
from app.ingestion.loader import Document

logger = structlog.get_logger(__name__)


@dataclass
class Chunk:
    """
    Represents a document chunk with text and metadata.
    
    Chunk IDs are deterministic based on content hash for idempotency.
    """
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")
    
    @property
    def page_number(self) -> Optional[int]:
        return self.metadata.get("page_number")
    
    @property
    def section(self) -> Optional[str]:
        return self.metadata.get("section")


class SemanticChunker:
    """
    Splits documents into semantic chunks with overlap.
    
    Chunking strategy:
    1. Split text into sentences
    2. Group sentences to reach target chunk size
    3. Add overlap from previous chunk
    4. Preserve all metadata from parent document
    
    Deterministic: Same input always produces same chunks.
    """
    
    # Sentence boundary pattern
    SENTENCE_PATTERN = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n+'
    )
    
    # Fallback split on any punctuation + space
    FALLBACK_PATTERN = re.compile(r'(?<=[.!?;:])\s+')
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize chunker with settings."""
        self._settings = settings or get_settings()
        self._chunk_size = self._settings.chunk_size
        self._chunk_overlap = self._settings.chunk_overlap
        
        logger.info(
            "Initialized SemanticChunker",
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
    
    def _generate_chunk_id(self, text: str, index: int, source: str) -> str:
        """
        Generate deterministic chunk ID.
        
        Based on content hash for idempotency - same content = same ID.
        """
        content = f"{source}:{index}:{text[:100]}"
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"chunk_{hash_val}"
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Try primary pattern
        sentences = self.SENTENCE_PATTERN.split(text)
        
        # Filter empty strings and strip
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If we got very few sentences, try fallback
        if len(sentences) <= 1 and len(text) > self._chunk_size:
            sentences = self.FALLBACK_PATTERN.split(text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # If still just one chunk and it's too long, split by paragraphs
        if len(sentences) <= 1 and len(text) > self._chunk_size:
            sentences = text.split('\n\n')
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # Last resort: split by newlines
        if len(sentences) <= 1 and len(text) > self._chunk_size:
            sentences = text.split('\n')
            sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count.
        
        Rough approximation: ~4 characters per token for English.
        """
        return len(text) // 4
    
    def chunk_document(self, document: Document) -> list[Chunk]:
        """
        Split a document into semantic chunks.
        
        Args:
            document: Document to chunk.
            
        Returns:
            List of Chunk objects with preserved metadata.
        """
        text = document.text.strip()
        
        if not text:
            return []
        
        # If text is small enough, return as single chunk
        if self._estimate_tokens(text) <= self._chunk_size:
            chunk_id = self._generate_chunk_id(text, 0, document.source)
            return [Chunk(
                chunk_id=chunk_id,
                text=text,
                metadata={
                    **document.metadata,
                    "chunk_index": 0,
                    "total_chunks": 1,
                },
            )]
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        # Group sentences into chunks
        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_tokens = 0
        overlap_sentences: list[str] = []
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            # Check if adding this sentence exceeds chunk size
            if current_tokens + sentence_tokens > self._chunk_size and current_sentences:
                # Create chunk from current sentences
                chunk_text = " ".join(current_sentences)
                chunk_id = self._generate_chunk_id(chunk_text, len(chunks), document.source)
                
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": len(chunks),
                    },
                ))
                
                # Calculate overlap sentences for next chunk
                overlap_sentences = self._get_overlap_sentences(
                    current_sentences, self._chunk_overlap
                )
                
                # Start new chunk with overlap
                current_sentences = overlap_sentences.copy()
                current_tokens = sum(self._estimate_tokens(s) for s in current_sentences)
            
            # Add sentence to current chunk
            current_sentences.append(sentence)
            current_tokens += sentence_tokens
        
        # Don't forget the last chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk_id = self._generate_chunk_id(chunk_text, len(chunks), document.source)
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                metadata={
                    **document.metadata,
                    "chunk_index": len(chunks),
                },
            ))
        
        # Update total_chunks in all metadata
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        
        logger.debug(
            "Chunked document",
            source=document.source,
            original_length=len(text),
            chunks=len(chunks),
        )
        
        return chunks
    
    def _get_overlap_sentences(
        self,
        sentences: list[str],
        overlap_tokens: int,
    ) -> list[str]:
        """Get sentences from end to include as overlap."""
        if not sentences or overlap_tokens <= 0:
            return []
        
        overlap: list[str] = []
        tokens = 0
        
        for sentence in reversed(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            if tokens + sentence_tokens > overlap_tokens:
                break
            overlap.insert(0, sentence)
            tokens += sentence_tokens
        
        return overlap
    
    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of documents to chunk.
            
        Returns:
            List of all chunks from all documents.
        """
        all_chunks: list[Chunk] = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(
            "Chunked documents batch",
            documents=len(documents),
            total_chunks=len(all_chunks),
        )
        
        return all_chunks


class FixedSizeChunker:
    """
    Alternative chunker that splits by character count.
    
    Useful for documents where sentence detection is unreliable.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        """Initialize with size parameters in characters."""
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
    
    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split document into fixed-size chunks."""
        text = document.text.strip()
        
        if not text:
            return []
        
        if len(text) <= self._chunk_size:
            chunk_id = f"chunk_{hashlib.sha256(text.encode()).hexdigest()[:12]}"
            return [Chunk(
                chunk_id=chunk_id,
                text=text,
                metadata={**document.metadata, "chunk_index": 0, "total_chunks": 1},
            )]
        
        chunks: list[Chunk] = []
        start = 0
        
        while start < len(text):
            end = start + self._chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                # Look for space near the end
                space_pos = text.rfind(' ', start + self._chunk_size - 100, end)
                if space_pos > start:
                    end = space_pos
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_id = f"chunk_{hashlib.sha256(chunk_text.encode()).hexdigest()[:12]}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": len(chunks),
                        "char_start": start,
                        "char_end": end,
                    },
                ))
            
            # Move start with overlap
            start = end - self._chunk_overlap
            if start <= chunks[-1].metadata.get("char_start", 0) if chunks else -1:
                start = end  # Prevent infinite loop
        
        # Update total chunks
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks
