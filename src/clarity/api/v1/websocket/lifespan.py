"""FastAPI lifespan management for WebSocket features."""

# removed - breaks FastAPI

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import inspect
import logging
from typing import Any

from fastapi import FastAPI

from clarity.api.v1.websocket.connection_manager import ConnectionManager

logger = logging.getLogger(__name__)

# Application state storage
connection_manager: ConnectionManager | None = None


def get_connection_manager() -> ConnectionManager:
    """Dependency to get the connection manager instance.

    This should be used as a FastAPI dependency:
    Uses app.state.connection_manager if available, else falls back to module-level singleton.
    """
    # Try to get from FastAPI app state if running in request context
    current_frame = inspect.currentframe()
    # One level up for the caller of get_connection_manager
    caller_frame = current_frame.f_back if current_frame is not None else None
    # Two levels up for the context where 'request' might be defined (e.g., the endpoint function)
    endpoint_frame = caller_frame.f_back if caller_frame is not None else None

    request_obj: Any | None = None

    if endpoint_frame and "request" in endpoint_frame.f_locals:
        request_obj = endpoint_frame.f_locals["request"]
    elif (
        caller_frame and "request" in caller_frame.f_locals
    ):  # Check caller frame as well
        request_obj = caller_frame.f_locals["request"]
    elif (
        current_frame and "request" in current_frame.f_locals
    ):  # Check current frame (less likely for typical use)
        request_obj = current_frame.f_locals["request"]

    if (
        request_obj
        and hasattr(request_obj, "app")
        and hasattr(request_obj.app.state, "connection_manager")
    ):
        manager = request_obj.app.state.connection_manager
        if isinstance(manager, ConnectionManager):
            return manager
        logger.warning(
            "app.state.connection_manager is not a ConnectionManager instance. Type: %s",
            type(manager),
        )

    # Fallback to module-level singleton
    global connection_manager  # noqa: PLW0603 - Module-level singleton pattern for WebSocket connection manager
    if connection_manager is not None:
        return connection_manager

    # For testing, try to get from test helper
    try:
        from tests.api.v1.test_websocket_helper import (  # noqa: PLC0415
            get_test_connection_manager,
        )

        # Ensure test manager is assigned to global if it's the one being used
        test_manager = get_test_connection_manager()
        if connection_manager is None:  # Assign if global is still None
            connection_manager = test_manager
        return test_manager
    except (ImportError, RuntimeError):
        pass

    # Last resort: create and assign to global
    logger.warning(
        "ConnectionManager not found in app.state or test helper, creating a new global instance."
    )
    # Note: background tasks won't be started in this fallback automatically
    new_manager = ConnectionManager(
        heartbeat_interval=5,
        max_connections_per_user=2,
        connection_timeout=30,
        message_rate_limit=10,
        max_message_size=1024,
    )
    connection_manager = new_manager  # Assign to the global variable
    return new_manager


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
    global connection_manager  # noqa: PLW0603 - Module-level singleton pattern for WebSocket connection manager

    # Startup: Create and initialize connection manager
    logger.info("Starting WebSocket services...")

    # Ensure a single ConnectionManager instance is created and used
    if connection_manager is None:
        connection_manager = ConnectionManager()

    await connection_manager.start_background_tasks()

    # Store in app state for dependency injection
    app.state.connection_manager = connection_manager

    logger.info("WebSocket services started successfully")

    try:
        yield
    finally:
        # Shutdown: Clean up connection manager
        logger.info("Shutting down WebSocket services...")

        # Use the manager from app.state if available, otherwise the global one
        manager_to_shutdown = getattr(
            app.state, "connection_manager", connection_manager
        )

        if manager_to_shutdown is not None:
            await manager_to_shutdown.shutdown()

        if hasattr(app.state, "connection_manager"):
            app.state.connection_manager = None

        connection_manager = None  # Clear the global reference

        logger.info("WebSocket services shut down successfully")
