"""
API routes initialization.
"""

from app.api.routes.ingest import router as ingest_router
from app.api.routes.query import router as query_router
from app.api.routes.health import router as health_router

__all__ = ["ingest_router", "query_router", "health_router"]
