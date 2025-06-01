"""CLARITY Digital Twin Platform - Main Application.

Production-grade FastAPI application for health data processing and AI-powered insights.
Implements enterprise authentication, monitoring, and HIPAA-compliant data handling.

Following Robert C. Martin's Clean Architecture with proper dependency injection.
"""

import logging

from fastapi import FastAPI
import uvicorn

from clarity.core.config import get_settings
from clarity.core.container import create_application as create_app_with_di
from clarity.core.logging_config import setup_logging

# Configure logging
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    DEPRECATED: Use create_application() from container.py instead.
    This function is kept for backward compatibility only.
    """
    logger.warning("Using deprecated create_app(). Use create_application() instead.")
    return create_application()


# Application factory following Clean Architecture principles
def create_application() -> FastAPI:
    """Application factory following Robert C. Martin's Clean Architecture.

    This factory function creates and configures the FastAPI application
    with all middleware, routes, and dependencies properly initialized.
    Follows Uncle Bob's SOLID principles and Gang of Four patterns.

    Uses dependency injection container for proper Clean Architecture.
    """
    # Use the DI container to create the application
    return create_app_with_di()


# Conditional app creation - only when explicitly requested
app: FastAPI | None = None


def get_application() -> FastAPI:
    """Get or create the FastAPI application instance.

    Implements lazy initialization pattern for better control
    over application lifecycle and testing. Follows Clean Code principles.

    Uses Clean Architecture dependency injection container.
    """
    global app
    if app is None:
        app = create_application()
    return app


# For production deployment and direct execution
if __name__ == "__main__":
    # Setup logging first
    settings = get_settings()
    setup_logging()

    # Create app using Clean Architecture container
    application = create_application()

    logger.info("Starting CLARITY Digital Twin Platform...")
    logger.info("Environment: %s", settings.environment)
    logger.info("Debug mode: %s", settings.debug)

    uvicorn.run(
        application,  # Pass app instance directly
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
else:
    # Module import - lazy creation for better testability
    app = get_application()
