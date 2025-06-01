"""CLARITY Digital Twin Platform - Main Application

Production-grade FastAPI application for health data processing and AI-powered insights.
Implements enterprise authentication, monitoring, and HIPAA-compliant data handling.
"""

from contextlib import asynccontextmanager
import logging
import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware
import uvicorn

from .api.v1 import health_data
from .auth import FirebaseAuthMiddleware
from .core.config import get_settings
from .core.logging_config import setup_logging

# Configure logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks."""
    # Startup
    logger.info("Starting CLARITY Digital Twin Platform...")

    # Initialize Firebase Admin SDK and other services here
    try:
        # Any additional startup initialization can go here
        logger.info("Application startup completed successfully")
        yield
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down CLARITY Digital Twin Platform...")
        # Cleanup tasks here
        logger.info("Application shutdown completed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Setup logging first
    setup_logging()

    # Get application settings
    settings = get_settings()

    # Create FastAPI application
    app = FastAPI(
        title="CLARITY Digital Twin Platform",
        description=(
            "AI-powered health data processing platform providing personalized insights "
            "through advanced machine learning and natural language processing."
        ),
        version="1.0.0",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        openapi_url="/openapi.json" if settings.environment != "production" else None,
        lifespan=lifespan
    )

    # Add security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )

    # Add CORS middleware
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

    # Add compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Add Firebase authentication middleware
    app.add_middleware(
        FirebaseAuthMiddleware,
        project_id=settings.firebase_project_id,
        exempt_paths=[
            "/",
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/api/v1/health-data/health"  # Public health check
        ],
        cache_ttl=300,  # 5 minutes
        enable_caching=True
    )

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Global exception handler for unhandled exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)

        if settings.environment == "production":
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_error",
                    "message": "An internal error occurred",
                    "request_id": getattr(request.state, "request_id", "unknown")
                }
            )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": str(exc),
                "type": type(exc).__name__,
                "request_id": getattr(request.state, "request_id", "unknown")
            }
        )

    # Health check endpoint
    @app.get(
        "/health",
        summary="Health Check",
        description="System health check endpoint for load balancers and monitoring."
    )
    async def health_check() -> dict[str, Any]:
        """System health check endpoint."""
        return {
            "status": "healthy",
            "service": "clarity-digital-twin",
            "version": "1.0.0",
            "environment": settings.environment,
            "timestamp": "2025-01-01T00:00:00Z"  # Should be dynamic
        }

    # Root endpoint
    @app.get(
        "/",
        summary="API Root",
        description="Root endpoint providing API information and status."
    )
    async def root() -> dict[str, Any]:
        """Root endpoint with API information."""
        return {
            "name": "CLARITY Digital Twin Platform",
            "version": "1.0.0",
            "description": "AI-powered health data processing and insights platform",
            "docs_url": "/docs" if settings.environment != "production" else None,
            "status": "operational"
        }

    # Include API routers
    app.include_router(
        health_data.router,
        prefix="/api/v1",
        tags=["Health Data API v1"]
    )

    logger.info(f"FastAPI application created successfully for {settings.environment} environment")
    return app


# Create the application instance
app = create_app()


# Development server entry point
if __name__ == "__main__":
    settings = get_settings()

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
        access_log=True
    )
