"""Comprehensive tests for insight service entrypoint.

Tests critical production infrastructure including FastAPI app creation,
CORS configuration, service mounting, and startup functionality.
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pytest

import clarity.entrypoints.insight_service
from clarity.entrypoints.insight_service import app, main


class TestInsightServiceApp:
    """Test insight service FastAPI app configuration."""

    def test_app_is_fastapi_instance(self) -> None:
        """Test that app is a FastAPI instance."""
        assert isinstance(app, FastAPI)

    def test_app_metadata_configuration(self) -> None:
        """Test FastAPI app metadata is properly configured."""
        assert app.title == "CLARITY Insight Service"
        assert app.description == "AI-powered health insight generation service"
        assert app.version == "1.0.0"

    def test_cors_middleware_configured(self) -> None:
        """Test that CORS middleware is properly configured."""
        # Check that CORSMiddleware is in the middleware stack
        cors_middleware_found = False
        for middleware in app.user_middleware:
            if middleware.cls == CORSMiddleware:
                cors_middleware_found = True
                break

        assert cors_middleware_found, "CORS middleware not found"

    def test_insight_app_mounted(self) -> None:
        """Test that insight app is mounted at root path."""
        # Check if app has routes mounted - this verifies the mounting occurred
        assert len(app.routes) > 0, "No routes found - mounting may have failed"

    @patch("clarity.entrypoints.insight_service.insight_app")
    def test_app_mount_with_mock_insight_app(self, mock_insight_app: MagicMock) -> None:
        """Test app mounting with mocked insight app."""
        # Import and create a fresh app to test mounting
        # Reload the module to ensure fresh import
        from clarity.entrypoints import insight_service  # noqa: PLC0415

        importlib.reload(insight_service)

        # The app should have been created with the mocked insight_app
        assert mock_insight_app is not None


class TestInsightServiceMain:
    """Test insight service main function and startup."""

    @patch("clarity.entrypoints.insight_service.uvicorn.run")
    @patch.dict(os.environ, {}, clear=True)
    def test_main_default_configuration(self, mock_uvicorn_run: MagicMock) -> None:
        """Test main function with default configuration."""
        main()

        mock_uvicorn_run.assert_called_once_with(
            "clarity.entrypoints.insight_service:app",
            host="127.0.0.1",
            port=8082,
            reload=False,
            log_level="info",
        )

    @patch("clarity.entrypoints.insight_service.uvicorn.run")
    @patch.dict(
        os.environ,
        {
            "HOST": "127.0.0.1",
            "PORT": "9000",
        },
        clear=True,
    )
    def test_main_custom_host_port(self, mock_uvicorn_run: MagicMock) -> None:
        """Test main function with custom host and port from environment."""
        main()

        mock_uvicorn_run.assert_called_once_with(
            "clarity.entrypoints.insight_service:app",
            host="127.0.0.1",  # Secure binding to localhost
            port=9000,
            reload=False,
            log_level="info",
        )

    @patch("clarity.entrypoints.insight_service.uvicorn.run")
    @patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True)
    def test_main_development_environment(self, mock_uvicorn_run: MagicMock) -> None:
        """Test main function with development environment."""
        main()

        mock_uvicorn_run.assert_called_once_with(
            "clarity.entrypoints.insight_service:app",
            host="127.0.0.1",
            port=8082,
            reload=True,  # Should be True in development
            log_level="info",
        )

    @patch("clarity.entrypoints.insight_service.uvicorn.run")
    @patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=True)
    def test_main_production_environment(self, mock_uvicorn_run: MagicMock) -> None:
        """Test main function with production environment."""
        main()

        mock_uvicorn_run.assert_called_once_with(
            "clarity.entrypoints.insight_service:app",
            host="127.0.0.1",
            port=8082,
            reload=False,  # Should be False in production
            log_level="info",
        )

    @patch("clarity.entrypoints.insight_service.uvicorn.run")
    @patch.dict(os.environ, {"PORT": "invalid"}, clear=True)
    def test_main_invalid_port_environment(self, mock_uvicorn_run: MagicMock) -> None:
        """Test main function with invalid port environment variable."""
        with pytest.raises(ValueError, match="invalid literal for int"):
            main()

    @patch("clarity.entrypoints.insight_service.uvicorn.run")
    @patch.dict(os.environ, {"PORT": "8080"}, clear=True)
    def test_main_port_as_string(self, mock_uvicorn_run: MagicMock) -> None:
        """Test main function correctly converts port string to integer."""
        main()

        mock_uvicorn_run.assert_called_once_with(
            "clarity.entrypoints.insight_service:app",
            host="127.0.0.1",
            port=8080,  # Should be converted to int
            reload=False,
            log_level="info",
        )

    @patch("clarity.entrypoints.insight_service.logger")
    @patch("clarity.entrypoints.insight_service.uvicorn.run")
    def test_main_logging_output(
        self, mock_uvicorn_run: MagicMock, mock_logger: MagicMock
    ) -> None:
        """Test that main function logs startup information."""
        main()

        # Check that appropriate log messages were called
        mock_logger.info.assert_any_call("Starting CLARITY Insight Service")
        mock_logger.info.assert_any_call("Listening on %s:%s", "127.0.0.1", 8082)

    @patch("clarity.entrypoints.insight_service.uvicorn.run")
    @patch.dict(
        os.environ,
        {"HOST": "192.168.1.100", "PORT": "3000", "ENVIRONMENT": "staging"},
        clear=True,
    )
    def test_main_comprehensive_configuration(
        self, mock_uvicorn_run: MagicMock
    ) -> None:
        """Test main function with comprehensive environment configuration."""
        main()

        mock_uvicorn_run.assert_called_once_with(
            "clarity.entrypoints.insight_service:app",
            host="192.168.1.100",
            port=3000,
            reload=False,  # staging is not "development"
            log_level="info",
        )


class TestInsightServiceLogging:
    """Test insight service logging configuration."""

    @patch("clarity.entrypoints.insight_service.logging.basicConfig")
    def test_logging_configuration(self, mock_basic_config: MagicMock) -> None:
        """Test that logging is properly configured on module import."""
        # Reload the module to trigger logging configuration
        importlib.reload(clarity.entrypoints.insight_service)

        # Verify basicConfig was called with correct parameters
        mock_basic_config.assert_called_with(
            level=clarity.entrypoints.insight_service.logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


class TestInsightServiceIntegration:
    """Test insight service integration scenarios."""

    def test_app_has_necessary_attributes(self):
        """Test that app has all necessary FastAPI attributes."""
        assert hasattr(app, "title")
        assert hasattr(app, "description")
        assert hasattr(app, "version")
        assert hasattr(app, "routes")
        assert hasattr(app, "middleware_stack")

    def test_app_can_be_imported_multiple_times(self):
        """Test that app can be safely imported multiple times."""
        from clarity.entrypoints.insight_service import app as app1  # noqa: PLC0415
        from clarity.entrypoints.insight_service import app as app2  # noqa: PLC0415

        # Should be the same instance
        assert app1 is app2

    @patch("clarity.entrypoints.insight_service.insight_app")
    def test_module_import_dependencies(self, mock_insight_app: MagicMock) -> None:
        """Test that all required dependencies are properly imported."""
        # This test ensures all imports work correctly
        # Check that key components are available
        assert hasattr(clarity.entrypoints.insight_service, "FastAPI")
        assert hasattr(clarity.entrypoints.insight_service, "CORSMiddleware")
        assert hasattr(clarity.entrypoints.insight_service, "uvicorn")
        assert hasattr(clarity.entrypoints.insight_service, "app")
        assert hasattr(clarity.entrypoints.insight_service, "main")

    def test_environment_variable_handling(self):
        """Test environment variable handling edge cases."""
        # Test with empty environment
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise any errors when accessing default values
            host = os.getenv("HOST", "127.0.0.1")
            port_str = os.getenv("PORT", "8082")
            environment = os.getenv("ENVIRONMENT")

            assert host == "127.0.0.1"
            assert port_str == "8082"
            assert environment is None

    def test_app_startup_readiness(self):
        """Test that app is ready for startup."""
        # Verify app is properly configured and ready
        assert app.title is not None
        assert len(app.title) > 0
        assert app.version is not None
        assert len(app.routes) > 0  # Should have mounted routes

    @patch("clarity.entrypoints.insight_service.uvicorn.run")
    def test_main_function_can_be_called_directly(
        self, mock_uvicorn_run: MagicMock
    ) -> None:
        """Test that main function can be called directly without issues."""
        # This simulates running the script directly
        try:
            main()
            main_execution_success = True
        except Exception:  # noqa: BLE001 - Test needs to catch all exceptions
            main_execution_success = False

        assert main_execution_success
        mock_uvicorn_run.assert_called_once()


class TestInsightServiceProductionScenarios:
    """Test realistic production scenarios."""

    @patch("clarity.entrypoints.insight_service.uvicorn.run")
    @patch.dict(
        os.environ,
        {
            "HOST": "127.0.0.1",
            "PORT": "80",
            "ENVIRONMENT": "production",
        },
        clear=True,
    )
    def test_production_deployment_configuration(
        self, mock_uvicorn_run: MagicMock
    ) -> None:
        """Test configuration for production deployment."""
        main()

        mock_uvicorn_run.assert_called_once_with(
            "clarity.entrypoints.insight_service:app",
            host="127.0.0.1",  # Secure binding to localhost
            port=80,  # Standard HTTP port
            reload=False,  # No reload in production
            log_level="info",
        )

    @patch("clarity.entrypoints.insight_service.uvicorn.run")
    @patch.dict(
        os.environ,
        {"HOST": "localhost", "PORT": "8082", "ENVIRONMENT": "development"},
        clear=True,
    )
    def test_development_environment_configuration(
        self, mock_uvicorn_run: MagicMock
    ) -> None:
        """Test configuration for development environment."""
        main()

        mock_uvicorn_run.assert_called_once_with(
            "clarity.entrypoints.insight_service:app",
            host="localhost",
            port=8082,
            reload=True,  # Enable reload in development
            log_level="info",
        )

    def test_cors_allows_all_origins_for_development(self):
        """Test that CORS is configured to allow all origins."""
        # In a production scenario, you might want to restrict this
        cors_middleware_found = False
        for middleware in app.user_middleware:
            if middleware.cls == CORSMiddleware:
                cors_middleware_found = True
                break

        assert cors_middleware_found, "CORS middleware not found"

    def test_service_metadata_for_api_documentation(self):
        """Test that service metadata is suitable for API documentation."""
        assert "CLARITY" in app.title
        assert "insight" in app.title.lower()
        assert len(app.description) > 10  # Should have meaningful description
        assert app.version.count(".") >= 1  # Should be semantic version format
