"""FastAPI lifespan management for WebSocket features."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from clarity.api.v1.websocket.connection_manager import ConnectionManager

logger = logging.getLogger(__name__)

# Application state storage
_connection_manager: ConnectionManager | None = None


def get_connection_manager() -> ConnectionManager:
    """Dependency to get the connection manager instance.

    This should be used as a FastAPI dependency:
    Uses app.state.connection_manager if available, else falls back to module-level singleton.
    """
    import sys
    from fastapi import Request

    # Try to get from FastAPI app state if running in request context
    frame = sys._getframe(1)
    local_vars = frame.f_locals
    if "request" in local_vars and hasattr(local_vars["request"], "app"):
        app = local_vars["request"].app
        if hasattr(app.state, "connection_manager"):
            return app.state.connection_manager

    # Fallback to module-level singleton
    if _connection_manager is not None:
        return _connection_manager

    # For testing, try to get from test helper
    try:
        from tests.api.v1.test_websocket_helper import get_test_connection_manager
        return get_test_connection_manager()
    except (ImportError, RuntimeError):
        pass

    # Last resort: create a test connection manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager(
            heartbeat_interval=5,
            max_connections_per_user=2,
            connection_timeout=30,
            message_rate_limit=10,
            max_message_size=1024,
        )
        # Note: background tasks won't be started in this fallback
    return _connection_manager


@asynccontextmanager
async def websocket_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager for WebSocket features.

    This should be used as the lifespan parameter when creating FastAPI apps
    that need WebSocket functionality:

    ```python
    from fastapi import FastAPI
    from .lifespan import websocket_lifespan

    app = FastAPI(lifespan=websocket_lifespan)
    ```

    The lifespan will:
    1. Create and initialize the connection manager during startup
    2. Start background tasks (heartbeat, cleanup)
    3. Store the manager in app state for dependency injection
    4. Clean up everything during shutdown
    """
    global _connection_manager

    # Startup: Create and initialize connection manager
    logger.info("Starting WebSocket services...")

    _connection_manager = ConnectionManager()
    await _connection_manager.start_background_tasks()

    # Store in app state for dependency injection
    app.state.connection_manager = _connection_manager

    logger.info("WebSocket services started successfully")

    try:
        yield
    finally:
        # Shutdown: Clean up connection manager
        logger.info("Shutting down WebSocket services...")

        if hasattr(app.state, "connection_manager"):
            await app.state.connection_manager.stop_background_tasks()
            app.state.connection_manager = None
        _connection_manager = None

        logger.info("WebSocket services shut down successfully")
