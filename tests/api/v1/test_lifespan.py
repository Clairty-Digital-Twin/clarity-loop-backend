from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from clarity.api.v1.websocket.connection_manager import ConnectionManager
from clarity.api.v1.websocket.lifespan import (
    get_connection_manager,
    websocket_lifespan,
)


def test_lifespan_startup_and_shutdown():
    """Test that the lifespan manager initializes and shuts down correctly."""
    with patch(
        "clarity.api.v1.websocket.lifespan.ConnectionManager", spec=ConnectionManager
    ) as mock_cm_constructor:
        mock_manager_instance = mock_cm_constructor.return_value
        mock_manager_instance.start_background_tasks = AsyncMock()
        mock_manager_instance.close_all = AsyncMock()

        app = FastAPI(lifespan=websocket_lifespan)

        with TestClient(app):
            # The lifespan startup should have been called
            mock_cm_constructor.assert_called_once()
            mock_manager_instance.start_background_tasks.assert_called_once()

        # The lifespan shutdown should have been called
        mock_manager_instance.close_all.assert_called_once()


def test_get_connection_manager_singleton():
    """Test that get_connection_manager returns a singleton instance."""
    manager = ConnectionManager()
    with patch("clarity.api.v1.websocket.lifespan.connection_manager", manager):
        get_connection_manager()
    # No direct assert, but we are checking if the call works as expected
    # The real test is that this doesn't create a new instance
    assert get_connection_manager() is manager
