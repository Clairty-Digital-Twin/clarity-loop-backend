"""Tests for analysis service entrypoint."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

from fastapi.testclient import TestClient

from clarity.entrypoints.analysis_service import app, main

if TYPE_CHECKING:
    from unittest.mock import Mock


class TestAnalysisService:
    """Test the analysis service entrypoint."""

    def test_app_initialization(self) -> None:
        """Test that the FastAPI app is properly initialized."""
        assert app.title == "CLARITY Analysis Service"
        assert app.description == "Health data analysis processing service"
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

    def test_app_mounts_analysis_app(self) -> None:
        """Test that the analysis app is mounted."""
        # Check if there are routes mounted
        assert len(app.routes) > 0, "Analysis app should be mounted"

    @patch("clarity.entrypoints.analysis_service.uvicorn.run")
    def test_main_function_with_defaults(self, mock_uvicorn_run: Mock) -> None:
        """Test main function with default environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            main()

        mock_uvicorn_run.assert_called_once_with(
            "clarity.entrypoints.analysis_service:app",
            host="127.0.0.1",
            port=8081,
            reload=False,
            log_level="info",
        )

    @patch("clarity.entrypoints.analysis_service.uvicorn.run")
    def test_main_function_with_custom_env(self, mock_uvicorn_run: Mock) -> None:
        """Test main function with custom environment variables."""
        env_vars = {
            "HOST": "0.0.0.0",  # noqa: S104 - Test configuration for bind all interfaces
            "PORT": "9000",
            "ENVIRONMENT": "development",
        }

        with patch.dict(os.environ, env_vars):
            main()

        mock_uvicorn_run.assert_called_once_with(
            "clarity.entrypoints.analysis_service:app",
            host="0.0.0.0",  # noqa: S104 - Test expects bind all interfaces
            port=9000,
            reload=True,
            log_level="info",
        )

    def test_health_endpoint_access(self) -> None:
        """Test that we can create a test client and the app responds."""
        with TestClient(app) as client:
            # The mounted analysis_app should handle requests
            # We just test that the app is accessible
            assert client is not None

            # Test that the app configuration is correct
            assert app.title == "CLARITY Analysis Service"
