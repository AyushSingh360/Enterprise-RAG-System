"""
Document ingestion endpoint.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from app.schemas.documents import IngestRequest, IngestResponse
from app.schemas.common import ErrorResponse
from app.api.dependencies import (
    get_ingestion_service,
    get_embedding_service,
    get_vector_store,
)
from app.ingestion.pipeline import IngestionService
from app.core.embeddings import EmbeddingService
from app.core.vector_store import VectorStoreService

router = APIRouter()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest Document",
    description="Ingest a document into the RAG knowledge base.",
    responses={
        200: {"model": IngestResponse},
        400: {
            "model": ErrorResponse,
            "description": "Invalid request or file not found"
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal processing error"
        }
    }
)
async def ingest_document(
    request: IngestRequest,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> IngestResponse | JSONResponse:
    """
    Ingest a document into the vector store.
    
    Process:
    1. Load document (PDF, DOCX, or Markdown)
    2. Split into semantic chunks
    3. Generate embeddings
    4. Store in FAISS vector store
    
    Args:
        request: Ingestion request with document info.
        
    Returns:
        IngestResponse with ingestion results.
    """
    response = await ingestion_service.ingest(
        request=request,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )
    
    if not response.success:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error="IngestionError",
                message=response.message,
                timestamp=datetime.now(timezone.utc),
            ).model_dump(mode="json"),
        )
    
    return response
