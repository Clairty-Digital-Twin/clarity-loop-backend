"""Configuration Provider Implementation.
"""Configuration provider for CLARITY application settings.

Provides a clean interface for accessing configuration values
with proper type safety and fallback handling.
"""

from typing import Any

from clarity.core.config import Settings
from clarity.core.interfaces import IConfigProvider


class ConfigProvider(IConfigProvider):
    """Configuration provider implementing IConfigProvider interface.

    Provides type-safe access to application settings with proper defaults.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize with settings instance."""
        self._settings = settings

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get configuration setting by key.

        Args:
            key: Configuration key to retrieve
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return getattr(self._settings, key, default)

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self._settings.environment == "development"

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self._settings.environment == "production"

    def get_database_url(self) -> str:
        """Get database connection URL."""
        return self._settings.database_url

    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        return self._settings.redis_url

    def should_skip_external_services(self) -> bool:
        """Determine if external services should be skipped.

        Returns True in development by default, False in production unless explicitly set.
        """
        # Skip in development mode by default
        if self.is_development():
            return bool(self.get_setting("skip_external_services", default=True))

        # In production, only skip if explicitly requested
        return bool(self.get_setting("skip_external_services", default=False))

    def get_database_url(self) -> str:
        """Get database connection URL with environment-specific defaults."""
        return self._settings.database_url

    def get_log_level(self) -> str:
        """Get logging level configuration."""
        return self._settings.log_level

    def get_firebase_project_id(self) -> str:
        """Get Firebase project ID."""
        return self._settings.firebase_project_id

    def get_firebase_credentials_path(self) -> str:
        """Get Firebase credentials file path."""
        return self._settings.firebase_credentials_path

    def get_cors_origins(self) -> list[str]:
        """Get CORS allowed origins list."""
        return self._settings.cors_origins

    def get_jwt_secret_key(self) -> str:
        """Get JWT secret key for token signing."""
        return self._settings.jwt_secret_key

    def get_jwt_algorithm(self) -> str:
        """Get JWT algorithm for token verification."""
        return self._settings.jwt_algorithm

    def get_jwt_access_token_expire_minutes(self) -> int:
        """Get JWT access token expiration time in minutes."""
        return self._settings.jwt_access_token_expire_minutes

    def get_app_name(self) -> str:
        """Get application name."""
        return self._settings.app_name

    def get_app_version(self) -> str:
        """Get application version."""
        return self._settings.app_version
