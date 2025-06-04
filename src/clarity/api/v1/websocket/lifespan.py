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
    if _connection_manager is None:
        raise RuntimeError(
            "Connection manager not initialized. "
            "Make sure FastAPI app uses the websocket lifespan."
        )
    return _connection_manager


@asynccontextmanager
async def websocket_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager for WebSocket features.
    
    This handles startup and shutdown of WebSocket-related resources:
    - Creates and starts the connection manager
    - Starts background tasks for heartbeats and cleanup
    - Gracefully shuts down during app termination
    
    Usage:
    ```python
    from fastapi import FastAPI
    from .websocket.lifespan import websocket_lifespan
    
    app = FastAPI(lifespan=websocket_lifespan)
    ```
    """
    global _connection_manager
    
    # Startup
    logger.info("Starting WebSocket services...")
    
    try:
        # Create connection manager
        _connection_manager = ConnectionManager()
        
        # Start background tasks
        await _connection_manager.start_background_tasks()
        
        # Store in app state for additional access if needed
        app.state.connection_manager = _connection_manager
        
        logger.info("WebSocket services started successfully")
        
        yield
        
    finally:
        # Shutdown
        logger.info("Shutting down WebSocket services...")
        
        if _connection_manager is not None:
            await _connection_manager.shutdown()
            _connection_manager = None
            
        if hasattr(app.state, 'connection_manager'):
            delattr(app.state, 'connection_manager')
            
        logger.info("WebSocket services shutdown complete")