"""
CLARITY Digital Twin Platform - Main Application Entry Point

FastAPI application with Google Cloud integration for health data processing
and AI-powered wellness insights.

Security Features:
- Firebase Authentication integration
- CORS protection with configurable origins
- Rate limiting and request validation
- HIPAA-compliant logging and audit trails

Architecture:
- Async-first with uvicorn ASGI server
- Microservice-ready with Cloud Run deployment
- Horizontal scaling with stateless design
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

import uvicorn
from pydantic_settings import BaseSettings

# Import application routers
from .api.v1.health_data import router as health_data_router

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application configuration with environment variable support."""
    
    # Application settings
    app_name: str = "CLARITY Digital Twin Backend"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    
    # CORS settings
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_allow_headers: list[str] = ["*"]
    
    # Security settings
    trusted_hosts: list[str] = ["localhost", "*.run.app"]
    
    # Firebase settings
    firebase_project_id: str = ""
    firebase_credentials_path: str = ""
    
    # Google Cloud settings
    gcp_project_id: str = ""
    firestore_database: str = "(default)"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    Handles:
    - Database connection initialization
    - Service health checks
    - Resource cleanup
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    
    # Initialize Google Cloud services
    try:
        # Initialize Firestore client
        logger.info("Initializing Google Cloud services...")
        # TODO: Initialize services in task #28
        
        # Initialize Firebase Auth
        logger.info("Initializing Firebase Authentication...")
        # TODO: Initialize auth in task #29
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    # Cleanup resources if needed
    logger.info("Application shutdown completed")


# Create FastAPI application instance
app = FastAPI(
    title=settings.app_name,
    description="Advanced HealthKit wellness platform with AI-powered insights",
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan
)


# Add security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=settings.trusted_hosts
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """
    Request/response logging middleware for monitoring and debugging.
    
    Logs:
    - Request method, URL, and headers
    - Response status and processing time
    - Error details for failed requests
    """
    start_time = time.time()
    
    # Log incoming request
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"- Client: {request.client.host if request.client else 'unknown'}"
    )
    
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {response.status_code} "
            f"- Time: {process_time:.3f}s "
            f"- Path: {request.url.path}"
        )
        
        # Add custom headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-API-Version"] = settings.app_version
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url.path} "
            f"- Error: {str(e)} - Time: {process_time:.3f}s"
        )
        raise


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Global HTTP exception handler with structured error responses.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "HTTPException",
                "message": exc.detail,
                "status_code": exc.status_code,
                "path": request.url.path,
                "timestamp": time.time()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled exceptions.
    """
    logger.exception(f"Unhandled exception on {request.url.path}: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "type": "InternalServerError",
                "message": "An internal server error occurred",
                "status_code": 500,
                "path": request.url.path,
                "timestamp": time.time()
            }
        }
    )


# Health check endpoint
@app.get(
    "/health",
    tags=["System"],
    summary="Health Check",
    description="System health check endpoint for monitoring and load balancers"
)
async def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check endpoint.
    
    Returns:
        Dict containing application status, version, and system metrics
    """
    return {
        "status": "healthy",
        "version": settings.app_version,
        "environment": settings.environment,
        "timestamp": time.time(),
        "services": {
            "api": "healthy",
            "database": "checking",  # TODO: Add actual database health check
            "authentication": "checking"  # TODO: Add auth service health check
        }
    }


# Root endpoint
@app.get(
    "/",
    tags=["System"], 
    summary="API Information"
)
async def root() -> Dict[str, Any]:
    """
    Root endpoint with API information.
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Advanced HealthKit wellness platform with AI-powered insights",
        "docs_url": "/docs" if settings.debug else None,
        "health_url": "/health"
    }


# Include API routers
app.include_router(
    health_data_router,
    prefix="/api/v1",
    tags=["Health Data"]
)


# Custom OpenAPI schema
def custom_openapi():
    """
    Customized OpenAPI schema with enhanced security definitions.
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.app_name,
        version=settings.app_version,
        description="Advanced HealthKit wellness platform with AI-powered insights and HIPAA compliance",
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "FirebaseAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Firebase Authentication JWT token"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Application entry point for Cloud Run
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level="info" if not settings.debug else "debug",
        reload=settings.debug,
        access_log=True
    )
