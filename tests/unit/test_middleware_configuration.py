"""Test middleware configuration functionality.

Tests that the middleware configuration system works correctly across
different environments (development, testing, production) as specified
in subtask 29.2.
"""

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from clarity.core.config import MiddlewareConfig, Settings
from clarity.core.config_provider import ConfigProvider
from clarity.core.container import create_application


class TestMiddlewareConfiguration:
    """Test suite for middleware configuration functionality."""

    def test_middleware_config_development_defaults(self) -> None:
        """Test middleware configuration defaults for development environment."""
        settings = Settings()
        settings.environment = "development"
        settings.enable_auth = True
        config_provider = ConfigProvider(settings)

        middleware_config = config_provider.get_middleware_config()

        # Development should have permissive settings
        assert middleware_config.enabled is True
        assert middleware_config.graceful_degradation is True
        assert middleware_config.fallback_to_mock is True
        assert middleware_config.log_successful_auth is True
        assert middleware_config.cache_enabled is False  # Disabled for easier debugging
        assert middleware_config.initialization_timeout_seconds == 10  # Longer timeout

    def test_middleware_config_testing_defaults(self) -> None:
        """Test middleware configuration defaults for testing environment."""
        settings = Settings()
        settings.environment = "testing"
        settings.enable_auth = True
        config_provider = ConfigProvider(settings)

        middleware_config = config_provider.get_middleware_config()

        # Testing should use mock auth
        assert middleware_config.enabled is False  # Usually mock in tests
        assert middleware_config.graceful_degradation is True
        assert middleware_config.fallback_to_mock is True
        assert middleware_config.log_successful_auth is False
        assert middleware_config.cache_enabled is False  # Consistent tests
        assert middleware_config.audit_logging is False  # Reduce noise

    def test_middleware_config_production_defaults(self) -> None:
        """Test middleware configuration defaults for production environment."""
        settings = Settings()
        settings.environment = "production"
        settings.enable_auth = True
        # Mock required production settings to avoid validation errors
        settings.firebase_project_id = "test-project"
        settings.gcp_project_id = "test-gcp-project"
        settings.firebase_credentials_path = "/test/path"

        config_provider = ConfigProvider(settings)

        middleware_config = config_provider.get_middleware_config()

        # Production should have strict settings
        assert middleware_config.enabled is True
        assert middleware_config.graceful_degradation is False  # Fail fast
        assert middleware_config.fallback_to_mock is False  # No mock fallback
        assert middleware_config.log_successful_auth is False  # Only log failures
        assert middleware_config.cache_enabled is True  # Performance
        assert middleware_config.cache_ttl_seconds == 600  # Longer cache
        assert middleware_config.initialization_timeout_seconds == 5  # Shorter timeout

    def test_middleware_config_exempt_paths_default(self) -> None:
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

    def test_middleware_config_custom_exempt_paths(self) -> None:
        """Test custom exempt paths configuration."""
        custom_paths = ["/custom", "/api/public"]
        config = MiddlewareConfig(exempt_paths=custom_paths)

        assert config.exempt_paths == custom_paths

    def test_config_provider_middleware_methods(self) -> None:
        """Test config provider middleware-specific methods."""
        settings = Settings(environment="development")
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

    def test_container_uses_middleware_config(self) -> None:
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

    def test_middleware_config_with_auth_disabled(self) -> None:
        """Test middleware configuration when auth is disabled."""
        settings = Settings(environment="production", enable_auth=False)
        config_provider = ConfigProvider(settings)

        middleware_config = config_provider.get_middleware_config()

        # When auth is disabled, middleware should reflect production settings
        # but the global enable_auth flag controls overall auth state
        assert (
            middleware_config.enabled is True
        )  # This is just the middleware capability
        assert not settings.enable_auth  # But global auth is disabled

    def test_cache_configuration_parameters(self) -> None:
        """Test cache configuration parameters."""
        config = MiddlewareConfig(
            cache_enabled=True,
            cache_ttl_seconds=1800,  # 30 minutes
            cache_max_size=2000,
        )

        assert config.cache_enabled is True
        assert config.cache_ttl_seconds == 1800
        assert config.cache_max_size == 2000
