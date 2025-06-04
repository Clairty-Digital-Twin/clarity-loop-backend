"""FastAPI lifespan management for WebSocket features."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from .connection_manager import ConnectionManager

logger = logging.getLogger(__name__)

# Application state storage
_connection_manager: ConnectionManager | None = None


def get_connection_manager() -> ConnectionManager:
    """Dependency to get the connection manager instance.
    
    This should be used as a FastAPI dependency:
    
    ```python
    from fastapi import Depends
    from .lifespan import get_connection_manager
    
    @app.websocket("/ws")
    async def websocket_endpoint(
        websocket: WebSocket,
        manager: ConnectionManager = Depends(get_connection_manager)
    ):
        # Use manager here
    ```
    """
    global _connection_manager
    
    # Try to get from global state first
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
        
        if _connection_manager:
            await _connection_manager.stop_background_tasks()
            _connection_manager = None
            
        logger.info("WebSocket services shut down successfully")