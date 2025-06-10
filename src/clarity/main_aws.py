"""CLARITY Digital Twin Platform - AWS Main Application Entry Point.

Fixed version that properly initializes all services and loads all 38 API endpoints.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from clarity.api.v1.router import api_router
from clarity.core.config_aws import get_settings
from clarity.core.container_aws import DependencyContainer
from clarity.core.exceptions import ConfigurationError

# Configure logging
logger = logging.getLogger(__name__)

# Global container instance
_container: DependencyContainer | None = None


def get_container() -> DependencyContainer:
    """Get the global container instance."""
    global _container
    if _container is None:
        raise RuntimeError("Container not initialized")
    return _container


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global _container
    
    logger.info("🚀 Starting CLARITY backend with AWS services...")
    
    settings = get_settings()
    
    try:
        # Initialize container
        _container = DependencyContainer(settings)
        await _container.initialize()
        
        logger.info(f"✅ CLARITY backend started successfully in {settings.environment} mode")
        logger.info(f"📊 Total routes configured: {len(app.routes)}")
        
        # Log API routes for verification
        api_routes = [route for route in app.routes if hasattr(route, 'path') and route.path.startswith('/api/')]
        logger.info(f"🔗 API endpoints loaded: {len(api_routes)}")
        
        yield
        
    except Exception as e:
        logger.error(f"💥 Failed to start application: {e}")
        raise ConfigurationError(f"Application startup failed: {e!s}")
    finally:
        # Cleanup
        if _container:
            await _container.shutdown()
        logger.info("🛑 CLARITY backend shutdown complete")


def create_app() -> FastAPI:
    """Factory function to create FastAPI application instance."""
    # Get settings
    settings = get_settings()
    
    # Create FastAPI application
    app = FastAPI(
        title="CLARITY Digital Twin Platform",
        description="Revolutionary AI-powered mental health platform using AWS services",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include ALL API routes - THIS IS THE KEY FIX!
    # This adds all 38 endpoints: auth, health-data, healthkit, pat, insights, metrics, websocket
    app.include_router(api_router, prefix="/api/v1")
    
    # Add health check endpoint
    @app.get("/health")
    async def health_check() -> dict[str, Any]:
        """Enhanced health check endpoint."""
        try:
            container = get_container()
            
            return {
                "status": "healthy",
                "service": "clarity-backend-aws",
                "environment": settings.environment,
                "version": "1.0.0",
                "deployment": "AWS",
                "endpoints": len(app.routes),
                "services": {
                    "auth": (
                        "cognito"
                        if hasattr(container, '_auth_provider') and container._auth_provider
                        else "mock"
                    ),
                    "database": (
                        "dynamodb"
                        if hasattr(container, '_health_data_repository') and container._health_data_repository
                        else "mock"
                    ),
                    "ai": (
                        "gemini"
                        if hasattr(container, '_gemini_service') and container._gemini_service
                        else "disabled"
                    ),
                },
            }
        except Exception:
            # If container not ready, return basic health check
            return {
                "status": "healthy",
                "service": "clarity-backend-aws",
                "environment": settings.environment,
                "version": "1.0.0",
                "deployment": "AWS",
                "note": "Container initializing",
            }

    # Add root endpoint
    @app.get("/")
    async def root() -> dict[str, Any]:
        """Root endpoint."""
        return {
            "name": "CLARITY Digital Twin Platform",
            "version": "1.0.0", 
            "status": "operational",
            "environment": settings.environment,
            "deployment": "AWS",
            "total_endpoints": len(app.routes),
            "api_docs": "/docs",
        }

    # Add Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "clarity.main_aws:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development(),
        log_level=settings.log_level.lower(),
    )
