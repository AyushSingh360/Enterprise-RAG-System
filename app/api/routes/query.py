"""
Query endpoint for the RAG system.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.schemas.query import QueryRequest, QueryResponse
from app.schemas.common import ErrorResponse
from app.api.dependencies import get_retriever_service
from app.core.retriever import RetrieverService

router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Query Documents",
    description="Ask a question and get an answer based solely on ingested documents."
)
async def query_documents(
    request: QueryRequest,
    retriever: RetrieverService = Depends(get_retriever_service)
) -> QueryResponse:
    """
    Query the RAG system with a question.
    
    The system will:
    1. Convert your question to an embedding
    2. Search for relevant document chunks
    3. Generate an answer using ONLY the retrieved context
    4. Return the answer with source citations
    
    If no relevant context is found, responds with "Answer not found in documents."
    """
    try:
        response = await retriever.query(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )
