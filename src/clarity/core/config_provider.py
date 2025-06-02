"""Configuration Provider Implementation.

Following Clean Architecture and SOLID principles, this module provides
concrete implementation of IConfigProvider interface for dependency injection.
"""

from typing import Any

from clarity.core.config import Settings
from clarity.core.interfaces import IConfigProvider


class ConfigProvider(IConfigProvider):
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
        """Check if running in development mode.

        Returns:
            True if in development environment, False otherwise
        """
        return self._settings.environment.lower() == "development"

    def should_skip_external_services(self) -> bool:
        """Check if external services should be skipped.

        Skip external services in development mode or when explicitly configured.
        This prevents startup hangs when Firebase/Firestore credentials are missing.

        Returns:
            True if external services should be skipped, False otherwise
        """
        # Skip in development mode by default
        if self.is_development():
            return bool(self.get_setting("skip_external_services", True))

        # In production, only skip if explicitly requested
        return bool(self.get_setting("skip_external_services", False))

    def get_database_url(self) -> str:
        """Get database connection URL.

        Returns:
            Database connection URL
        """
        return getattr(self._settings, "database_url", "")

    def get_firebase_config(self) -> dict[str, Any]:
        """Get Firebase configuration.

        Returns:
            Firebase configuration dictionary
        """
        return {
            "project_id": getattr(self._settings, "firebase_project_id", ""),
            "credentials_path": getattr(self._settings, "firebase_credentials", ""),
            "web_api_key": getattr(self._settings, "firebase_web_api_key", ""),
        }

    def is_auth_enabled(self) -> bool:
        """Check if authentication is enabled.

        Returns:
            True if authentication should be enabled, False otherwise
        """
        return getattr(self._settings, "enable_auth", True)

    def get_gcp_project_id(self) -> str:
        """Get Google Cloud Platform project ID.

        Returns:
            GCP project ID
        """
        return getattr(self._settings, "gcp_project_id", "")

    def get_log_level(self) -> str:
        """Get logging level.

        Returns:
            Log level string
        """
        return getattr(self._settings, "log_level", "INFO")
