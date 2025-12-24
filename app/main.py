"""
Main FastAPI application entry point.
"""

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.config import get_settings
from app.api.routes.health import router as health_router
from app.api.routes.ingest import router as ingest_router
from app.api.routes.query import router as query_router

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    settings = get_settings()
    logger.info(
        "Starting Enterprise RAG System",
        version=__version__,
        environment=settings.app_env
    )
    yield
    logger.info("Shutting down Enterprise RAG System")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Enterprise RAG System",
        description="""
## Enterprise Retrieval-Augmented Generation System

A production-grade RAG system that answers questions **strictly from private data**.

### Key Features:
- **No Hallucination**: Answers are generated ONLY from ingested documents
- **Source Citation**: Every answer includes document sources
- **Multi-format Support**: PDF, DOCX, Markdown, and SQL data
- **Semantic Search**: FAISS-powered similarity search with configurable thresholds

### Usage:
1. **Ingest Documents**: Use `/ingest` to add documents to the knowledge base
2. **Query**: Use `/query` to ask questions
3. **Health Check**: Use `/health` to verify system status
        """,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "details": str(exc) if not settings.is_production else None
            }
        )
    
    # Include routers
    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(query_router)
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
