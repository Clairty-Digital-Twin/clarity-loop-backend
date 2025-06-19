"""Configuration Provider Implementation.

Following Clean Architecture and SOLID principles, this module provides
concrete implementation of IConfigProvider interface for dependency injection.
"""

# removed - breaks FastAPI

import os
from typing import TYPE_CHECKING

from clarity.core.config_aws import MiddlewareConfig, Settings
from clarity.ports.config_ports import IConfigProvider

if TYPE_CHECKING:
    pass  # Only for type stubs now


class ConfigProvider(IConfigProvider):  # noqa: PLR0904 - Feature-rich config provider
    """Concrete implementation of configuration provider.

    Follows Single Responsibility Principle - only handles configuration access.
    Implements Dependency Inversion Principle by depending on Settings abstraction.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize configuration provider with settings.

        Args:
            settings: Configuration settings object
        """
        self._settings = settings

    def get_setting(
        self, key: str, *, default: str | int | bool | None = None
    ) -> str | int | bool | None:
        """Get configuration setting by key.

        Args:
            key: Configuration key to retrieve
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return getattr(self._settings, key, default)

    def is_development(self) -> bool:
        """Check if running in development mode.

        Returns:
            True if in development environment, False otherwise
        """
        return self._settings.environment == "development"

    def is_testing(self) -> bool:
        """Check if running in testing mode.

        Returns:
            True if in testing environment, False otherwise
        """
        return self._settings.environment == "testing"

    def is_production(self) -> bool:
        """Check if running in production mode.

        Returns:
            True if in production environment, False otherwise
        """
        return self._settings.environment == "production"

    def should_skip_external_services(self) -> bool:
        """Check if external services should be skipped.

        Skip external services in development mode, testing mode, or when explicitly configured.
        This prevents startup hangs when cloud credentials are missing.

        Returns:
            True if external services should be skipped, False otherwise
        """
        return self.is_development() or self.is_testing()

    def get_database_url(self) -> str:
        """Get database connection URL.

        Returns:
            Database connection URL
        """
        # Return AWS-based URL
        return f"https://{self.get_aws_region()}.amazonaws.com"

    def get_dynamodb_url(self) -> str:
        """Get DynamoDB service URL."""
        return f"https://dynamodb.{self.get_aws_region()}.amazonaws.com"

    def get_aws_config(self) -> dict[str, str]:
        """Get AWS configuration."""
        return {
            "region": self.get_aws_region(),
            "cognito_user_pool_id": os.getenv("COGNITO_USER_POOL_ID", ""),
            "dynamodb_table": os.getenv("DYNAMODB_TABLE_NAME", "clarity-health-data"),
        }

    def is_auth_enabled(self) -> bool:
        """Check if authentication is enabled.

        Returns:
            True if authentication should be enabled, False otherwise
        """
        return getattr(self._settings, "enable_auth", False)

    def get_aws_region(self) -> str:
        """Get AWS region from settings."""
        region = self.get_setting("aws_region")
        if region:
            return str(region)  # Ensure it's a string

        # Fallback to default if not configured
        return "us-east-1"

    def get_log_level(self) -> str:
        """Get logging level.

        Returns:
            Log level string
        """
        return getattr(self._settings, "log_level", "INFO")

    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        return getattr(self._settings, "redis_url", "redis://localhost:6379")

    def get_cors_origins(self) -> list[str]:
        """Get CORS allowed origins list."""
        origins = getattr(self._settings, "cors_origins", "")
        if origins:
            return [origin.strip() for origin in origins.split(",")]
        return ["http://localhost:3000", "http://localhost:8080"]

    def get_jwt_secret_key(self) -> str:
        """Get JWT secret key for token signing."""
        return getattr(self._settings, "jwt_secret_key", "development-secret-key")

    def get_jwt_algorithm(self) -> str:
        """Get JWT algorithm for token verification."""
        return getattr(self._settings, "jwt_algorithm", "HS256")

    def get_jwt_access_token_expire_minutes(self) -> int:
        """Get JWT access token expiration time in minutes."""
        return getattr(self._settings, "jwt_access_token_expire_minutes", 30)

    def get_app_name(self) -> str:
        """Get application name."""
        return getattr(self._settings, "app_name", "CLARITY Digital Twin")

    def get_app_version(self) -> str:
        """Get application version."""
        return getattr(self._settings, "app_version", "1.0.0")

    def get_api_base_url(self) -> str:
        """Get API base URL."""
        return getattr(self._settings, "api_base_url", "http://localhost:8000")

    def get_rate_limit_settings(self) -> dict[str, int]:
        """Get rate limiting settings."""
        return {
            "requests_per_minute": getattr(self._settings, "rate_limit_rpm", 100),
            "burst_size": getattr(self._settings, "rate_limit_burst", 10),
        }

    def get_middleware_config(self) -> MiddlewareConfig:
        """Get environment-specific middleware configuration.

        Returns:
            MiddlewareConfig: Configuration object with environment-specific settings
        """
        return self._settings.middleware_config

    def get_auth_timeout_seconds(self) -> int:
        """Get authentication timeout in seconds.

        Returns:
            int: Timeout for auth operations
        """
        middleware_config = self.get_middleware_config()
        return middleware_config.initialization_timeout_seconds

    def should_enable_auth_cache(self) -> bool:
        """Check if authentication token caching should be enabled.

        Returns:
            bool: True if caching should be enabled
        """
        middleware_config = self.get_middleware_config()
        return middleware_config.cache_enabled

    def get_auth_cache_ttl(self) -> int:
        """Get authentication cache TTL in seconds.

        Returns:
            int: Cache TTL in seconds
        """
        middleware_config = self.get_middleware_config()
        return middleware_config.cache_ttl_seconds

    def get_settings_model(self) -> Settings:
        """Get the settings model instance.

        Returns:
            Settings: The settings model instance
        """
        return self._settings
