"""
Query endpoint for RAG retrieval and generation.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
import structlog

from app.schemas.query import QueryRequest, QueryResponse
from app.schemas.common import ErrorResponse
from app.api.dependencies import get_retriever_service
from app.core.retriever import RetrieverService

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query Documents",
    description="""
Query the RAG system with a question.

**Process:**
1. Embed query
2. Retrieve relevant document chunks (with similarity threshold)
3. If no relevant context found → return "Answer not found in documents"
4. Generate answer using ONLY retrieved context
5. Return answer with **mandatory source citations**

**Important:**
- The `sources` array is ALWAYS included
- If answer is found, `sources` contains citations
- If no answer found, `sources` is empty and confidence is 0.0
    """,
    responses={
        200: {
            "model": QueryResponse,
            "description": "Query response with answer and citations"
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid query request"
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal processing error"
        }
    }
)
async def query_documents(
    request: QueryRequest,
    retriever: RetrieverService = Depends(get_retriever_service),
) -> QueryResponse | JSONResponse:
    """
    Query the knowledge base.
    
    The response ALWAYS includes:
    - `answer`: The generated answer or "Answer not found in documents."
    - `sources`: Array of SourceCitation (empty if no answer)
    - `confidence`: 0.0-1.0 (0.0 means no answer found)
    
    Args:
        request: Query request with question and parameters.
        
    Returns:
        QueryResponse with answer and mandatory citations.
    """
    try:
        response = await retriever.query(request)
        return response
        
    except ValueError as e:
        logger.warning("Query validation error", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error="ValidationError",
                message=str(e),
                timestamp=datetime.now(timezone.utc),
            ).model_dump(mode="json"),
        )
    except Exception as e:
        logger.error("Query processing error", error=str(e), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="QueryError",
                message="Failed to process query",
                timestamp=datetime.now(timezone.utc),
            ).model_dump(mode="json"),
        )
