"""Example FastAPI app integration with WebSocket lifespan management.

This file demonstrates how to properly integrate the WebSocket connection manager
with FastAPI's lifespan events for production use.
"""

# removed - breaks FastAPI

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from clarity.api.v1.websocket.chat_handler import router as chat_router
from clarity.api.v1.websocket.lifespan import get_connection_manager, websocket_lifespan

if TYPE_CHECKING:
    pass  # Only for type stubs now


@asynccontextmanager
async def full_app_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Complete app lifespan that can be extended with other services.

    This example shows how to combine WebSocket lifespan with other
    application startup/shutdown logic.
    """
    # You can add other startup logic here

    # Use websocket_lifespan as a nested context manager
    async with websocket_lifespan(app):
        yield


def create_websocket_app() -> FastAPI:
    """Create FastAPI app with WebSocket support.

    This is the recommended way to create a FastAPI app that uses
    the WebSocket connection manager with proper lifecycle management.

    Returns:
        FastAPI: Configured FastAPI application
    """
    app = FastAPI(
        title="WebSocket Chat API",
        description="Real-time chat with health insights",
        version="1.0.0",
        lifespan=websocket_lifespan,  # Use websocket lifespan
    )

    # Include WebSocket routes
    app.include_router(chat_router, prefix="/api/v1")

    @app.get("/")
    async def root() -> dict[str, str]:
        """Health check endpoint."""
        return {"message": "WebSocket Chat API is running"}

    @app.get("/health")
    async def health_check() -> dict[str, str | dict[str, int | str]]:
        """Detailed health check with WebSocket status."""
        # Import at top-level to comply with linting; if circular import occurs, move back with justification.
        try:
            connection_manager = get_connection_manager()
            return {
                "status": "healthy",
                "websocket": {
                    "active_connections": connection_manager.get_connection_count(),
                    "active_users": connection_manager.get_user_count(),
                },
            }
        except RuntimeError:
            return {
                "status": "starting",
                "websocket": {"status": "initializing"},
            }

    return app


def create_extended_app() -> FastAPI:
    """Create FastAPI app with extended lifespan management.

    Use this pattern when you need to add other services alongside
    WebSocket functionality.

    Returns:
        FastAPI: Configured FastAPI application with extended lifespan
    """
    app = FastAPI(
        title="Extended WebSocket API",
        description="WebSocket API with additional services",
        version="1.0.0",
        lifespan=full_app_lifespan,  # Use extended lifespan
    )

    # Include WebSocket routes
    app.include_router(chat_router, prefix="/api/v1")

    @app.get("/")
    async def root() -> dict[str, str]:
        """Health check endpoint."""
        return {"message": "Extended WebSocket API is running"}

    return app


# For development/testing
if __name__ == "__main__":
    import uvicorn

    app = create_websocket_app()

    # For local development, bind to localhost only for security (see S104 warning)
    uvicorn.run(
        app,
        host="127.0.0.1",  # Changed from 0.0.0.0 to 127.0.0.1 for dev/test safety
        port=8000,
        log_level="info",
        reload=False,  # Set to True for development
    )
