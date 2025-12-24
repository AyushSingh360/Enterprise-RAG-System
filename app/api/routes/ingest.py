"""
Document ingestion endpoint.
"""

from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import shutil
from pathlib import Path

from app.schemas.documents import (
    DocumentType,
    IngestRequest,
    IngestResponse,
)
from app.schemas.common import ErrorResponse
from app.api.dependencies import get_ingestion_pipeline
from app.ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post(
    "",
    response_model=IngestResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Ingest Document",
    description="Ingest a document into the RAG system. Supports PDF, DOCX, Markdown, and SQL."
)
async def ingest_document(
    request: IngestRequest,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline)
) -> IngestResponse:
    """
    Ingest a document into the vector store.
    
    The document will be:
    1. Loaded from the specified source
    2. Chunked into semantic segments
    3. Embedded using OpenAI embeddings
    4. Stored in the FAISS vector store
    """
    response = await pipeline.ingest(request)
    
    if not response.success:
        return JSONResponse(
            status_code=400,
            content=response.model_dump()
        )
    
    return response


@router.post(
    "/file",
    response_model=IngestResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Upload and Ingest File",
    description="Upload a file directly and ingest it into the RAG system."
)
async def ingest_file(
    file: UploadFile = File(..., description="File to ingest (PDF or DOCX)"),
    metadata: Optional[str] = Form(default=None, description="JSON metadata"),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline)
) -> IngestResponse:
    """
    Upload and ingest a file directly.
    """
    import json
    
    # Determine document type from extension
    filename = file.filename or "document"
    ext = Path(filename).suffix.lower()
    
    type_map = {
        ".pdf": DocumentType.PDF,
        ".docx": DocumentType.DOCX,
        ".md": DocumentType.MARKDOWN,
    }
    
    if ext not in type_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: .pdf, .docx, .md"
        )
    
    doc_type = type_map[ext]
    
    # Parse metadata if provided
    meta = {}
    if metadata:
        try:
            meta = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON metadata")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        # Create request based on type
        if doc_type == DocumentType.MARKDOWN:
            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()
            request = IngestRequest(
                document_type=doc_type,
                content=content,
                metadata={"source": filename, **meta}
            )
        else:
            request = IngestRequest(
                document_type=doc_type,
                file_path=tmp_path,
                metadata={"source": filename, **meta}
            )
        
        response = await pipeline.ingest(request)
        
        if not response.success:
            return JSONResponse(status_code=400, content=response.model_dump())
        
        return response
        
    finally:
        Path(tmp_path).unlink(missing_ok=True)
