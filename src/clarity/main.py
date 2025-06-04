"""CLARITY Digital Twin Platform - Main Application Entry Point.

FastAPI application setup with Clean Architecture dependency injection.
This module serves as the composition root for the entire application.
"""

import logging

from fastapi import FastAPI, WebSocket
import uvicorn

from clarity.core.config import get_settings
from clarity.core.container import create_application
from clarity.core.logging_config import setup_logging

# Configure logging
logger = logging.getLogger(__name__)


# Application factory function
def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Factory function for creating the application instance.
    Uses Clean Architecture dependency injection container.
    """
    return create_application()


# For development and testing
def get_app() -> FastAPI:
    """Get application instance for development/testing.

    Creates a new instance each time to avoid global state issues.
    Uses Clean Architecture dependency injection container.
    """
    return create_application()


# For production deployment (uvicorn/gunicorn)
app = create_application()


# Add WebSocket route
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")


# For production deployment and direct execution
if __name__ == "__main__":
    # Setup logging first
    settings = get_settings()
    setup_logging()

    logger.info("Starting CLARITY Digital Twin Platform...")
    logger.info("Environment: %s", settings.environment)
    logger.info("Debug mode: %s", settings.debug)

    uvicorn.run(
        app,  # Pass app instance directly
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
else:
    # Module import - lazy creation for better testability
    app = get_app()
