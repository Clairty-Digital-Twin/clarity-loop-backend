"""CLARITY Digital Twin Platform - AWS Main Application Entry Point."""

from contextlib import asynccontextmanager
import logging
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from clarity.core.config_aws import get_settings
from clarity.core.container_aws import get_container, initialize_container
from clarity.core.exceptions import ConfigurationError

# Configure logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting CLARITY backend with AWS services...")

    settings = get_settings()

    try:
        # Initialize dependency container
        container = await initialize_container(settings)

        # Configure routes
        container.configure_routes(app)

        logger.info(
            f"CLARITY backend started successfully in {settings.environment} mode"
        )

        yield

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise ConfigurationError(f"Application startup failed: {e!s}")
    finally:
        # Cleanup
        container = get_container()
        await container.shutdown()
        logger.info("CLARITY backend shutdown complete")


def create_app() -> FastAPI:
    """Factory function to create FastAPI application instance.
    
    This function is used by tests and follows the same pattern as the original
    CLARITY implementation, allowing the test suite to work with the AWS version.
    """
    # Create FastAPI application
    app = FastAPI(
        title="CLARITY Digital Twin Platform",
        description="Revolutionary AI-powered mental health platform using AWS services",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Configure CORS
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    @app.get("/")
    async def root() -> dict[str, Any]:
        """Root endpoint."""
        return {
            "name": "CLARITY Digital Twin Platform",
            "version": "1.0.0",
            "status": "operational",
            "environment": settings.environment,
            "deployment": "AWS",
        }
    
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    # For local development only
    uvicorn.run(
        "clarity.main_aws:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development(),
        log_level=settings.log_level.lower(),
    )
