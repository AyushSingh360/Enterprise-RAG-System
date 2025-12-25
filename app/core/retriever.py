"""
RAG Retriever: Orchestrates the complete retrieval-augmented generation pipeline.

CRITICAL FLOW:
1. RETRIEVE documents FIRST (before any generation)
2. ENFORCE similarity threshold
3. If no relevant context → return "Answer not found in documents"
4. Generate answer using ONLY retrieved context
5. Include source citations in response
"""

import time
from typing import Optional, Any
from dataclasses import dataclass
import structlog

from app.config import Settings, get_settings
from app.core.embeddings import EmbeddingService
from app.core.vector_store import VectorStoreService, SearchResult
from app.core.llm import LLMService
from app.schemas.query import (
    QueryRequest,
    QueryResponse,
    SourceCitation,
    RetrievedContext,
)

logger = structlog.get_logger(__name__)


@dataclass
class RetrievalResult:
    """Internal result from retrieval phase."""
    contexts: list[str]
    source_labels: list[str]
    citations: list[SourceCitation]
    retrieved_contexts: list[RetrievedContext]
    retrieval_time_ms: float


class RetrieverService:
    """
    Orchestrates the complete RAG pipeline.
    
    Pipeline flow:
    1. Embed query
    2. Retrieve relevant chunks from vector store
    3. Apply similarity threshold (reject low-score results)
    4. If no results pass threshold → return "Answer not found"
    5. Generate answer from context ONLY
    6. Return answer with mandatory citations
    """
    
    NO_ANSWER_MESSAGE = "Answer not found in documents."
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStoreService] = None,
        llm_service: Optional[LLMService] = None,
    ):
        """Initialize the retriever with all required services."""
        self._settings = settings or get_settings()
        self._embedding_service = embedding_service or EmbeddingService(self._settings)
        self._vector_store = vector_store or VectorStoreService(self._settings)
        self._llm_service = llm_service or LLMService(self._settings)
        
        logger.info(
            "Initialized RetrieverService",
            top_k=self._settings.top_k,
            similarity_threshold=self._settings.similarity_threshold,
        )
    
    async def _retrieve(
        self,
        query: str,
        top_k: Optional[int],
        similarity_threshold: Optional[float],
        metadata_filter: Optional[dict[str, Any]],
    ) -> RetrievalResult:
        """
        PHASE 1: Retrieve relevant documents.
        
        This MUST happen BEFORE generation.
        Enforces similarity threshold to filter irrelevant results.
        """
        start_time = time.perf_counter()
        
        # Step 1: Generate query embedding
        query_embedding = await self._embedding_service.get_query_embedding(query)
        
        # Step 2: Search vector store with threshold
        threshold = similarity_threshold or self._settings.similarity_threshold
        k = top_k or self._settings.top_k
        
        search_results: list[SearchResult] = await self._vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=k,
            similarity_threshold=threshold,
            metadata_filter=metadata_filter,
        )
        
        retrieval_time = (time.perf_counter() - start_time) * 1000
        
        # Step 3: Extract context and build citations
        contexts: list[str] = []
        source_labels: list[str] = []
        citations: list[SourceCitation] = []
        retrieved_contexts: list[RetrievedContext] = []
        
        for result in search_results:
            chunk = result.chunk
            score = result.score
            
            # Add to context list
            contexts.append(chunk.text)
            
            # Build source label
            label_parts = [chunk.source]
            if chunk.page_number:
                label_parts.append(f"page {chunk.page_number}")
            if chunk.section:
                label_parts.append(f"section: {chunk.section}")
            source_label = ", ".join(label_parts)
            source_labels.append(source_label)
            
            # Build citation
            citation = SourceCitation(
                document_id=chunk.document_id,
                source=chunk.source,
                page_number=chunk.page_number,
                section=chunk.section,
                relevance_score=round(score, 4),
                chunk_text=chunk.text[:500] if len(chunk.text) > 500 else chunk.text,
            )
            citations.append(citation)
            
            # Build retrieved context
            retrieved_context = RetrievedContext(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                similarity_score=round(score, 4),
                metadata={
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                },
            )
            retrieved_contexts.append(retrieved_context)
        
        logger.info(
            "Retrieval complete",
            query_length=len(query),
            results_count=len(search_results),
            threshold=threshold,
            retrieval_time_ms=round(retrieval_time, 2),
        )
        
        return RetrievalResult(
            contexts=contexts,
            source_labels=source_labels,
            citations=citations,
            retrieved_contexts=retrieved_contexts,
            retrieval_time_ms=retrieval_time,
        )
    
    def _create_no_answer_response(
        self,
        retrieval_time_ms: float = 0.0,
        generation_time_ms: float = 0.0,
        total_time_ms: float = 0.0,
    ) -> QueryResponse:
        """Create response when no relevant context is found."""
        return QueryResponse(
            success=True,
            answer=self.NO_ANSWER_MESSAGE,
            sources=[],  # No sources = no answer
            context=None,
            confidence=0.0,
            query_time_ms=round(total_time_ms, 2),
            retrieval_time_ms=round(retrieval_time_ms, 2),
            generation_time_ms=round(generation_time_ms, 2),
        )
    
    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a query through the complete RAG pipeline.
        
        FLOW:
        1. RETRIEVE first (mandatory)
        2. Check if any results pass threshold
        3. If no results → "Answer not found in documents"
        4. Generate answer from context ONLY
        5. Return with citations
        
        Args:
            request: Query request with question and parameters.
            
        Returns:
            QueryResponse with answer, sources, and timing.
        """
        start_time = time.perf_counter()
        
        question = request.question.strip()
        
        logger.info(
            "Processing query",
            question_length=len(question),
            top_k=request.top_k,
            threshold=request.similarity_threshold,
        )
        
        # =====================================================
        # PHASE 1: RETRIEVAL (MUST happen before generation)
        # =====================================================
        
        retrieval_result = await self._retrieve(
            query=question,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            metadata_filter=request.metadata_filter,
        )
        
        # =====================================================
        # CHECK: No relevant context found
        # =====================================================
        
        if not retrieval_result.contexts:
            logger.info(
                "No relevant context found",
                question=question[:100],
                threshold=request.similarity_threshold or self._settings.similarity_threshold,
            )
            total_time = (time.perf_counter() - start_time) * 1000
            return self._create_no_answer_response(
                retrieval_time_ms=retrieval_result.retrieval_time_ms,
                total_time_ms=total_time,
            )
        
        # =====================================================
        # PHASE 2: GENERATION (with context ONLY)
        # =====================================================
        
        generation_start = time.perf_counter()
        
        answer, confidence = await self._llm_service.generate_answer(
            question=question,
            context_texts=retrieval_result.contexts,
            source_labels=retrieval_result.source_labels,
        )
        
        generation_time = (time.perf_counter() - generation_start) * 1000
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Check if LLM indicated no answer
        if confidence == 0.0:
            return self._create_no_answer_response(
                retrieval_time_ms=retrieval_result.retrieval_time_ms,
                generation_time_ms=generation_time,
                total_time_ms=total_time,
            )
        
        # =====================================================
        # BUILD RESPONSE WITH CITATIONS
        # =====================================================
        
        response = QueryResponse(
            success=True,
            answer=answer,
            sources=retrieval_result.citations,  # MANDATORY citations
            context=retrieval_result.retrieved_contexts if request.include_context else None,
            confidence=round(confidence, 4),
            query_time_ms=round(total_time, 2),
            retrieval_time_ms=round(retrieval_result.retrieval_time_ms, 2),
            generation_time_ms=round(generation_time, 2),
        )
        
        logger.info(
            "Query processed successfully",
            answer_length=len(answer),
            sources_count=len(retrieval_result.citations),
            confidence=confidence,
            total_time_ms=round(total_time, 2),
        )
        
        return response
    
    # =====================================================
    # Service Accessors (for dependency injection)
    # =====================================================
    
    @property
    def embedding_service(self) -> EmbeddingService:
        """Get embedding service."""
        return self._embedding_service
    
    @property
    def vector_store(self) -> VectorStoreService:
        """Get vector store service."""
        return self._vector_store
    
    @property
    def llm_service(self) -> LLMService:
        """Get LLM service."""
        return self._llm_service
