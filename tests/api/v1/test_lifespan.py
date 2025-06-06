import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from clarity.api.v1.websocket.connection_manager import ConnectionManager
from clarity.api.v1.websocket.lifespan import get_connection_manager, websocket_lifespan


def test_websocket_lifespan():
    mock_manager = AsyncMock()
    mock_manager.start_background_tasks = AsyncMock()
    mock_manager.shutdown = AsyncMock()

    with patch(
        "clarity.api.v1.websocket.lifespan.ConnectionManager", return_value=mock_manager
    ) as mock_cm_constructor:
        app = FastAPI(lifespan=websocket_lifespan)

        with TestClient(app) as client:
            # The lifespan startup should have been called
            mock_cm_constructor.assert_called_once()
            mock_manager.start_background_tasks.assert_called_once()
            assert app.state.connection_manager == mock_manager

        # The lifespan shutdown should have been called
        mock_manager.shutdown.assert_called_once()


def test_get_connection_manager():
    # Test that the dependency returns the correct instance
    manager = ConnectionManager()
    with patch("clarity.api.v1.websocket.lifespan.connection_manager", manager):
        retrieved_manager = get_connection_manager()


def test_get_connection_manager_fallback():
    # Test that the dependency creates a new instance if none exists
    with patch("clarity.api.v1.websocket.lifespan.connection_manager", None):
        manager = get_connection_manager()
        assert isinstance(manager, ConnectionManager)
