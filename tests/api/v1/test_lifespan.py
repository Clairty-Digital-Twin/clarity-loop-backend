"""Tests for WebSocket lifespan management."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from clarity.api.v1.websocket.connection_manager import ConnectionManager
import clarity.api.v1.websocket.lifespan as lifespan_module
from clarity.api.v1.websocket.lifespan import get_connection_manager, websocket_lifespan


def test_lifespan_startup_and_shutdown():
    """Test that the lifespan manager initializes and shuts down correctly."""
    mock_manager = MagicMock(spec=ConnectionManager)
    mock_manager.start_background_tasks = AsyncMock()
    mock_manager.shutdown = AsyncMock()

    # Patch the global variable to None initially
    with (
        patch("clarity.api.v1.websocket.lifespan.connection_manager", None),
        patch(
            "clarity.api.v1.websocket.lifespan.ConnectionManager",
            return_value=mock_manager,
        ),
    ):
        app = FastAPI(lifespan=websocket_lifespan)
        with TestClient(app) as client:
            # Startup should be called
            assert client
            mock_manager.start_background_tasks.assert_called_once()

        # Shutdown should be called
        mock_manager.shutdown.assert_called_once()


def test_get_connection_manager_singleton():
    """Test that get_connection_manager returns a singleton instance."""
    # Reset the global variable and patch ConnectionManager
    original_manager = lifespan_module.connection_manager

    try:
        lifespan_module.connection_manager = None
        manager1 = get_connection_manager()
        manager2 = get_connection_manager()
        assert manager1 is manager2
        assert isinstance(manager1, ConnectionManager)
    finally:
        # Restore original state
        lifespan_module.connection_manager = original_manager
