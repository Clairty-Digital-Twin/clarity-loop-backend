"""CLARITY Digital Twin Platform - Main Application.

Production-grade FastAPI application for health data processing and AI-powered insights.
Implements enterprise authentication, monitoring, and HIPAA-compliant data handling.
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import Response
import uvicorn

from clarity.api.v1 import health_data
from clarity.auth import FirebaseAuthMiddleware
from clarity.core.config import get_settings
from clarity.core.logging_config import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown tasks."""
    # Startup
    try:
        logger.info("Starting CLARITY Digital Twin Platform...")

        # Initialize any async resources here
        # (database connections, external service clients, etc.)
        await asyncio.sleep(0)  # Make function properly async

        logger.info("Application startup completed successfully")
        yield
    except Exception:
        logger.exception("Application startup failed")
        raise
    finally:
        # Cleanup
        logger.info("Application shutdown completed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = get_settings()

    # Create FastAPI app with lifespan management
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI-powered health data processing and insights platform for "
            "personalized healthcare and research applications."
        ),
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
    )

    # Configure CORS middleware
    if settings.environment != "production":
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts,
    )

    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Firebase Authentication middleware
    app.add_middleware(
        FirebaseAuthMiddleware,
        credentials_path=settings.firebase_credentials_path,
        project_id=settings.firebase_project_id,
        exempt_paths=[
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
        ],
    )

    # Health check endpoints
    @app.get("/")
    async def root() -> dict[str, Any]:
        """Root endpoint with API information."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "description": "AI-powered health data processing and insights platform",
            "docs_url": "/docs",
            "status": "operational",
        }

    @app.get("/health")
    async def health_check() -> dict[str, Any]:
        """Health check endpoint for monitoring and load balancers."""
        return {
            "status": "healthy",
            "service": "clarity-digital-twin",
            "version": settings.app_version,
            "environment": settings.environment,
            "timestamp": "2025-01-01T00:00:00Z",  # This should be dynamic
        }

    # Global exception handler
    @app.exception_handler(Exception)
    def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Global exception handler for unhandled exceptions."""
        logger.error("Unhandled exception: %s", exc)

        if settings.environment == "production":
            # In production, don't expose internal error details
            error_detail = "An internal error occurred. Please try again later."
        else:
            # In development, include more details for debugging
            error_detail = str(exc)

        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": error_detail,
                "path": str(request.url),
            },
        )

    # Include API routers
    app.include_router(
        health_data.router,
        prefix="/api/v1",
        tags=["Health Data"],
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Log HTTP requests for monitoring and debugging."""
        response = await call_next(request)

        # Log request details
        logger.info(
            "Request: %s %s - Response: %s",
            request.method,
            request.url.path,
            response.status_code,
        )

        return response

    # Configure OpenAPI documentation
    if settings.debug:
        app.openapi_tags = [
            {
                "name": "Health Data",
                "description": "Operations for health data upload, processing, and retrieval",
            },
            {
                "name": "Authentication",
                "description": "Firebase-based authentication and authorization",
            },
        ]

    env_msg = "FastAPI application created successfully for %s environment"
    logger.info(env_msg, settings.environment)
    return app


# Application instance
app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "src.clarity.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
