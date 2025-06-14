"""Helper utilities for WebSocket testing."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from clarity.api.v1.websocket.chat_handler import router as chat_router
from clarity.api.v1.websocket.connection_manager import ConnectionManager

# Global test connection manager
_test_connection_manager: ConnectionManager | None = None

MSG_MANAGER_NOT_INITIALIZED = "Test connection manager not initialized"


def get_test_connection_manager() -> ConnectionManager:
    """Get the test connection manager instance."""
    if _test_connection_manager is None:
        msg = MSG_MANAGER_NOT_INITIALIZED
        raise RuntimeError(msg)
    return _test_connection_manager


@asynccontextmanager
async def websocket_test_lifespan(app: FastAPI):
    """Lifespan function for WebSocket tests."""
    global _test_connection_manager  # noqa: PLW0603 - Test fixture state management

    # Create and start connection manager
    _test_connection_manager = ConnectionManager(
        heartbeat_interval=5,
        max_connections_per_user=2,
        connection_timeout=30,
        message_rate_limit=10,
        max_message_size=1024,
    )

    await _test_connection_manager.start_background_tasks()

    # Store in app state for dependency injection
    app.state.connection_manager = _test_connection_manager

    yield

    # Cleanup
    await _test_connection_manager.stop_background_tasks()
    _test_connection_manager = None


def create_websocket_test_app() -> FastAPI:
    """Create a FastAPI app specifically for WebSocket testing."""
    app = FastAPI(lifespan=websocket_test_lifespan)

    # Add WebSocket routes
    app.include_router(chat_router, prefix="/api/v1/ws")

    return app
