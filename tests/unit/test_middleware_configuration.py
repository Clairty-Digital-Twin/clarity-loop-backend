"""Test middleware configuration functionality.

Tests that the middleware configuration system works correctly across
different environments (development, testing, production) as specified
in subtask 29.2.
"""

from __future__ import annotations

import os
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from clarity.core.config_aws import MiddlewareConfig, Settings
from clarity.core.config_provider import ConfigProvider
from clarity.core.container import create_application


class TestMiddlewareConfiguration:
    """Test suite for middleware configuration functionality."""

    @staticmethod
    def test_middleware_config_development_defaults() -> None:
        """Test middleware configuration defaults for development environment."""
        # Use environment variable to ensure proper setting
        # Clear any existing environment variables that might interfere
        with patch.dict(os.environ, {}, clear=True):
            # Set only the variables we need
            os.environ["ENVIRONMENT"] = "development"
            os.environ["ENABLE_AUTH"] = "true"

            settings = Settings()
            config_provider = ConfigProvider(settings)

            middleware_config = config_provider.get_middleware_config()

            # Development should have permissive settings
            assert middleware_config.enabled is True
            assert (
                middleware_config.graceful_degradation is False
            )  # Show actual auth errors for debugging
            assert middleware_config.fallback_to_mock is True
            assert middleware_config.log_successful_auth is True
            assert (
                middleware_config.cache_enabled is False
            )  # Disabled for easier debugging
            assert (
                middleware_config.initialization_timeout_seconds == 10
            )  # Longer timeout

    @staticmethod
    def test_middleware_config_testing_defaults() -> None:
        """Test middleware configuration defaults for testing environment."""
        # Clear any existing environment variables that might interfere
        with patch.dict(os.environ, {}, clear=True):
            # Set only the variables we need
            os.environ["ENVIRONMENT"] = "testing"
            os.environ["ENABLE_AUTH"] = "true"

            settings = Settings()
            config_provider = ConfigProvider(settings)

            middleware_config = config_provider.get_middleware_config()

            # Testing should use mock auth
            assert middleware_config.enabled is True  # Follows enable_auth setting
            assert (
                middleware_config.graceful_degradation is False
            )  # Show actual auth errors
            assert middleware_config.fallback_to_mock is True
            assert middleware_config.log_successful_auth is False
            assert (
                middleware_config.cache_enabled is True
            )  # Default for unknown environments
            assert middleware_config.audit_logging is True  # Default value

    @staticmethod
    def test_middleware_config_production_defaults() -> None:
        """Test middleware configuration defaults for production environment."""
        # Use environment variables for production settings
        with patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "production",
                "ENABLE_AUTH": "true",
                "TESTING": "false",
                "AWS_REGION": "us-east-1",
                "COGNITO_USER_POOL_ID": "test-pool",
                "COGNITO_CLIENT_ID": "test-client",
            },
        ):
            settings = Settings()
            config_provider = ConfigProvider(settings)

            middleware_config = config_provider.get_middleware_config()

            # Production should have strict settings
            assert middleware_config.enabled is True
            assert middleware_config.graceful_degradation is False  # Fail fast
            assert middleware_config.fallback_to_mock is False  # No mock fallback
            assert middleware_config.log_successful_auth is False  # Only log failures
            assert middleware_config.cache_enabled is True  # Performance
            assert middleware_config.cache_ttl_seconds == 300  # 5 minutes
            assert (
                middleware_config.initialization_timeout_seconds == 5
            )  # Shorter timeout

    @staticmethod
    def test_middleware_config_exempt_paths_default() -> None:
        """Test that default exempt paths are properly configured."""
        config = MiddlewareConfig()

        expected_paths = [
            "/",
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/api/docs",
            "/api/health",
        ]

        assert config.exempt_paths == expected_paths

    @staticmethod
    def test_middleware_config_custom_exempt_paths() -> None:
        """Test custom exempt paths configuration."""
        custom_paths = ["/custom", "/api/public"]
        config = MiddlewareConfig(exempt_paths=custom_paths)

        assert config.exempt_paths == custom_paths

    @staticmethod
    def test_config_provider_middleware_methods() -> None:
        """Test config provider middleware-specific methods."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            settings = Settings()
            config_provider = ConfigProvider(settings)

            # Test timeout getter
            timeout = config_provider.get_auth_timeout_seconds()
            assert isinstance(timeout, int)
            assert timeout > 0

            # Test cache enabled getter
            cache_enabled = config_provider.should_enable_auth_cache()
            assert isinstance(cache_enabled, bool)

            # Test cache TTL getter
            cache_ttl = config_provider.get_auth_cache_ttl()
            assert isinstance(cache_ttl, int)
            assert cache_ttl > 0

    @staticmethod
    def test_container_uses_middleware_config() -> None:
        """Test that the container properly uses middleware configuration."""
        with patch("clarity.core.container.get_settings") as mock_get_settings:
            # Mock settings to disable auth for simpler testing
            mock_settings = Mock()
            mock_settings.environment = "testing"
            mock_settings.enable_auth = False
            mock_settings.get_middleware_config.return_value = MiddlewareConfig(
                enabled=False
            )
            mock_get_settings.return_value = mock_settings

            # Create application
            app = create_application()
            client = TestClient(app)

            # Test health endpoint works
            response = client.get("/health")
            assert response.status_code == 200

    @staticmethod
    def test_middleware_config_with_auth_disabled() -> None:
        """Test middleware configuration when auth is disabled."""
        with patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "production",
                "ENABLE_AUTH": "false",
                "TESTING": "false",
            },
        ):
            settings = Settings()
            config_provider = ConfigProvider(settings)

            middleware_config = config_provider.get_middleware_config()

            # When enable_auth is False, middleware.enabled should also be False
            assert middleware_config.enabled is False  # Follows enable_auth setting
            assert not settings.enable_auth  # Global auth is disabled

    @staticmethod
    def test_cache_configuration_parameters() -> None:
        """Test cache configuration parameters."""
        config = MiddlewareConfig(
            cache_enabled=True,
            cache_ttl_seconds=1800,  # 30 minutes
            cache_max_size=2000,
        )

        assert config.cache_enabled is True
        assert config.cache_ttl_seconds == 1800
        assert config.cache_max_size == 2000
