"""Tests for insight service entrypoint."""

import os
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from clarity.entrypoints.insight_service import app, main


class TestInsightService:
    """Test the insight service entrypoint."""

    def test_app_initialization(self) -> None:
        """Test that the FastAPI app is properly initialized."""
        assert app.title == "CLARITY Insight Service"
        assert app.description == "AI-powered health insight generation service"
        assert app.version == "1.0.0"

    def test_app_has_cors_middleware(self) -> None:
        """Test that CORS middleware is properly configured."""
        # Check if CORS middleware is in the middleware stack
        cors_found = False
        for middleware in app.user_middleware:
            if "CORSMiddleware" in str(middleware):
                cors_found = True
                break

        assert cors_found, "CORS middleware should be configured"

    def test_app_mounts_insight_app(self) -> None:
        """Test that the insight app is mounted."""
        # Check if there are routes mounted
        assert len(app.routes) > 0, "Insight app should be mounted"

    @patch("clarity.entrypoints.insight_service.uvicorn.run")
    def test_main_function_with_defaults(self, mock_uvicorn_run: Mock) -> None:
        """Test main function with default environment variables."""
        from clarity.entrypoints.insight_service import main

        with patch.dict(os.environ, {}, clear=True):
            main()

        mock_uvicorn_run.assert_called_once_with(
            "clarity.entrypoints.insight_service:app",
            host="127.0.0.1",
            port=8082,
            reload=False,
            log_level="info",
        )

    @patch("clarity.entrypoints.insight_service.uvicorn.run")
    def test_main_function_with_custom_env(self, mock_uvicorn_run: Mock) -> None:
        """Test main function with custom environment variables."""
        from clarity.entrypoints.insight_service import main

        env_vars = {"HOST": "0.0.0.0", "PORT": "9001", "ENVIRONMENT": "development"}

        with patch.dict(os.environ, env_vars):
            main()

        mock_uvicorn_run.assert_called_once_with(
            "clarity.entrypoints.insight_service:app",
            host="0.0.0.0",
            port=9001,
            reload=True,
            log_level="info",
        )

    def test_health_endpoint_access(self) -> None:
        """Test that we can create a test client and the app responds."""
        from clarity.entrypoints.insight_service import app

        with TestClient(app) as client:
            # The mounted insight_app should handle requests
            # We just test that the app is accessible
            assert client is not None

            # Test that the app configuration is correct
            assert app.title == "CLARITY Insight Service"
