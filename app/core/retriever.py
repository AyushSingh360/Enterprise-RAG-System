"""
Retriever service that combines embedding, vector search, and generation.
Orchestrates the complete RAG pipeline with source citation.
"""

import time
from typing import Optional, Any
import structlog

from llama_index.core.schema import NodeWithScore

from app.config import Settings, get_settings
from app.core.embeddings import EmbeddingService
from app.core.vector_store import VectorStoreService
from app.core.llm import LLMService
from app.schemas.query import (
    QueryRequest,
    QueryResponse,
    SourceCitation,
    RetrievedContext,
)

logger = structlog.get_logger(__name__)


class RetrieverService:
    """
    Main RAG retriever service that orchestrates the complete pipeline:
    1. Query embedding
    2. Similarity search
    3. Context extraction
    4. Answer generation with citations
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStoreService] = None,
        llm_service: Optional[LLMService] = None
    ):
        """
        Initialize the retriever service.
        
        Args:
            settings: Application settings.
            embedding_service: Service for generating embeddings.
            vector_store: Service for vector storage and search.
            llm_service: Service for answer generation.
        """
        self._settings = settings or get_settings()
        self._embedding_service = embedding_service or EmbeddingService(self._settings)
        self._vector_store = vector_store or VectorStoreService(
            self._settings, self._embedding_service
        )
        self._llm_service = llm_service or LLMService(self._settings)
    
    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            request: The query request containing the question and parameters.
            
        Returns:
            QueryResponse with answer, sources, and timing information.
        """
        start_time = time.perf_counter()
        
        # Step 1: Generate query embedding
        retrieval_start = time.perf_counter()
        query_embedding = await self._embedding_service.get_query_embedding(
            request.question
        )
        
        # Step 2: Perform similarity search
        results = await self._vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=request.top_k or self._settings.top_k,
            similarity_threshold=request.similarity_threshold or self._settings.similarity_threshold,
            metadata_filter=request.metadata_filter
        )
        
        retrieval_end = time.perf_counter()
        retrieval_time_ms = (retrieval_end - retrieval_start) * 1000
        
        # Step 3: Check if we have relevant results
        if not results:
            logger.info(
                "No relevant context found",
                question=request.question[:100]
            )
            return self._create_no_answer_response(
                retrieval_time_ms=retrieval_time_ms,
                total_time_ms=(time.perf_counter() - start_time) * 1000
            )
        
        # Step 4: Extract context and citations
        context_texts, source_citations = self._extract_context_and_citations(results)
        
        # Step 5: Generate answer
        generation_start = time.perf_counter()
        answer, confidence = await self._llm_service.generate_answer(
            question=request.question,
            context_texts=context_texts,
            source_citations=[c.source for c in source_citations]
        )
        generation_end = time.perf_counter()
        generation_time_ms = (generation_end - generation_start) * 1000
        
        # Step 6: Build response
        total_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Check if answer indicates no result
        if self._is_no_answer(answer):
            return self._create_no_answer_response(
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=generation_time_ms,
                total_time_ms=total_time_ms
            )
        
        # Build successful response
        response = QueryResponse(
            success=True,
            answer=answer,
            sources=source_citations,
            context=self._build_context(results) if request.include_context else None,
            confidence=confidence,
            query_time_ms=total_time_ms,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms
        )
        
        logger.info(
            "Query processed successfully",
            question_length=len(request.question),
            results_count=len(results),
            confidence=confidence,
            total_time_ms=total_time_ms
        )
        
        return response
    
    def _extract_context_and_citations(
        self,
        results: list[NodeWithScore]
    ) -> tuple[list[str], list[SourceCitation]]:
        """
        Extract context texts and build source citations from search results.
        
        Args:
            results: List of NodeWithScore from similarity search.
            
        Returns:
            Tuple of (context_texts, source_citations).
        """
        context_texts: list[str] = []
        source_citations: list[SourceCitation] = []
        
        for result in results:
            node = result.node
            score = result.score
            
            # Extract text
            text = node.get_content()
            context_texts.append(text)
            
            # Build citation
            metadata = node.metadata or {}
            citation = SourceCitation(
                document_id=metadata.get("document_id", "unknown"),
                source=metadata.get("source", metadata.get("file_name", "unknown")),
                page_number=metadata.get("page_number"),
                section=metadata.get("section"),
                relevance_score=score,
                chunk_text=text[:500] if len(text) > 500 else text
            )
            source_citations.append(citation)
        
        return context_texts, source_citations
    
    def _build_context(self, results: list[NodeWithScore]) -> list[RetrievedContext]:
        """Build RetrievedContext objects from search results."""
        contexts: list[RetrievedContext] = []
        
        for result in results:
            node = result.node
            metadata = node.metadata or {}
            
            context = RetrievedContext(
                chunk_id=metadata.get("node_id", node.node_id),
                document_id=metadata.get("document_id", "unknown"),
                text=node.get_content(),
                similarity_score=result.score,
                metadata=metadata
            )
            contexts.append(context)
        
        return contexts
    
    def _is_no_answer(self, answer: str) -> bool:
        """Check if the answer indicates no result found."""
        answer_lower = answer.lower()
        no_answer_indicators = [
            "answer not found in documents",
            "i don't know",
            "cannot find",
            "no information available",
            "not mentioned in the provided",
            "not found in the documents"
        ]
        return any(indicator in answer_lower for indicator in no_answer_indicators)
    
    def _create_no_answer_response(
        self,
        retrieval_time_ms: float = 0.0,
        generation_time_ms: float = 0.0,
        total_time_ms: float = 0.0
    ) -> QueryResponse:
        """Create a response for when no answer is found."""
        return QueryResponse(
            success=True,
            answer="Answer not found in documents.",
            sources=[],
            context=None,
            confidence=0.0,
            query_time_ms=total_time_ms,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms
        )
    
    @property
    def vector_store(self) -> VectorStoreService:
        """Get the vector store service."""
        return self._vector_store
    
    @property
    def embedding_service(self) -> EmbeddingService:
        """Get the embedding service."""
        return self._embedding_service
    
    @property
    def llm_service(self) -> LLMService:
        """Get the LLM service."""
        return self._llm_service
