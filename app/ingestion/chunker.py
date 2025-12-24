"""
Semantic chunking with overlap handling and metadata preservation.
"""

import uuid
from typing import Optional, Any
import structlog

from llama_index.core.schema import Document, TextNode
from llama_index.core.node_parser import SentenceSplitter

from app.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class SemanticChunker:
    """
    Semantic chunker that splits documents into meaningful chunks.
    Preserves metadata and handles overlap between chunks.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the chunker with settings."""
        self._settings = settings or get_settings()
        self._splitter = SentenceSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[.!?]\\s+",
        )
    
    def chunk_document(self, document: Document) -> list[TextNode]:
        """
        Split a document into semantic chunks.
        
        Args:
            document: The document to chunk.
            
        Returns:
            List of TextNode chunks with preserved metadata.
        """
        nodes = self._splitter.get_nodes_from_documents([document])
        
        for i, node in enumerate(nodes):
            node.metadata = {
                **document.metadata,
                "chunk_index": i,
                "total_chunks": len(nodes),
                "node_id": str(uuid.uuid4()),
            }
        
        logger.debug(
            "Chunked document",
            source=document.metadata.get("source", "unknown"),
            chunks=len(nodes)
        )
        return nodes
    
    def chunk_documents(self, documents: list[Document]) -> list[TextNode]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of documents to chunk.
            
        Returns:
            List of all TextNode chunks.
        """
        all_nodes: list[TextNode] = []
        
        for doc in documents:
            nodes = self.chunk_document(doc)
            all_nodes.extend(nodes)
        
        logger.info(
            "Chunked documents batch",
            documents=len(documents),
            total_chunks=len(all_nodes)
        )
        return all_nodes
