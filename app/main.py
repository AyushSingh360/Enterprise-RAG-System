"""
Main FastAPI application entry point.

This module initializes the FastAPI application with:
- Structured logging (JSON format for production)
- CORS middleware
- Router registration
- Health check endpoint
- Global exception handling
- Lifespan management
"""

import sys
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app import __version__
from app.config import get_settings, Settings
from app.api.routes.health import router as health_router
from app.api.routes.ingest import router as ingest_router
from app.api.routes.query import router as query_router


# ===================
# Structured Logging Configuration
# ===================

def configure_logging(settings: Settings) -> None:
    """
    Configure structured logging based on environment settings.
    
    - Production: JSON format for log aggregation
    - Development: Console format for readability
    """
    # Set root logger level
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
    )
    
    # Common processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Choose renderer based on format setting
    if settings.log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure formatter for stdlib logging
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    
    # Apply formatter to root handler
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)


# Get logger after configuration
logger = structlog.get_logger(__name__)


# ===================
# Application Lifespan
# ===================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    
    Handles startup and shutdown events:
    - Startup: Validate configuration, initialize services
    - Shutdown: Cleanup resources
    """
    settings = get_settings()
    
    # Configure logging first
    configure_logging(settings)
    
    # Startup
    logger.info(
        "Starting application",
        app_name=settings.app_name,
        version=__version__,
        environment=settings.app_env,
        log_level=settings.log_level,
    )
    
    # Validate required secrets
    try:
        settings.validate_required_secrets()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error("Configuration validation failed", error=str(e))
        if settings.is_production:
            raise
        else:
            logger.warning("Continuing in development mode without valid API key")
    
    # Log safe configuration (secrets masked)
    logger.debug("Application configuration", config=settings.get_safe_dict())
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down application", app_name=settings.app_name)


# ===================
# Application Factory
# ===================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured application instance.
    """
    settings = get_settings()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        description="""
## Enterprise Retrieval-Augmented Generation (RAG) System

A production-grade RAG system that answers questions **strictly from private data**.

### Core Principles:
- ❌ **No Hallucination**: LLM answers ONLY from ingested documents
- ✅ **Mandatory Citations**: Every answer includes document sources
- 🔒 **Context-Only**: If no relevant context, returns "Answer not found"

### Endpoints:
- `POST /ingest` - Ingest documents (PDF, DOCX, Markdown, SQL)
- `POST /query` - Ask questions against the knowledge base
- `GET /health` - System health check

### Supported Document Types:
- PDF files with page-level metadata
- DOCX files with section extraction
- Markdown content
- SQL database query results
        """,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        debug=settings.debug,
    )
    
    # ===================
    # Middleware
    # ===================
    
    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    
    # ===================
    # Exception Handlers
    # ===================
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors."""
        logger.warning(
            "Validation error",
            path=request.url.path,
            errors=exc.errors(),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "ValidationError",
                "message": "Request validation failed",
                "details": exc.errors(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle all unhandled exceptions."""
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
            error_type=type(exc).__name__,
            error_message=str(exc),
            exc_info=True,
        )
        
        # Don't expose internal errors in production
        detail = str(exc) if not settings.is_production else "An unexpected error occurred"
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": detail,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
    
    # ===================
    # Request Logging Middleware
    # ===================
    
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all incoming requests and their processing time."""
        import time
        import uuid
        
        request_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()
        
        # Add request context to logs
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        
        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )
        
        response = await call_next(request)
        
        process_time = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            process_time_ms=round(process_time, 2),
        )
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        
        return response
    
    # ===================
    # Register Routers
    # ===================
    
    # Health check router (always enabled)
    app.include_router(health_router, tags=["Health"])
    
    # Ingestion router
    app.include_router(ingest_router, tags=["Ingestion"])
    
    # Query router
    app.include_router(query_router, tags=["Query"])
    
    # ===================
    # Root Endpoint
    # ===================
    
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint redirecting to docs or returning API info."""
        return {
            "name": settings.app_name,
            "version": __version__,
            "status": "operational",
            "docs": "/docs" if not settings.is_production else "disabled",
            "health": "/health",
        }
    
    return app


# ===================
# Application Instance
# ===================

app = create_app()


# ===================
# Development Server
# ===================

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
        access_log=True,
    )
