"""Configuration port interfaces.

Defines the contract for configuration providers following Clean Architecture.
Business logic layer depends on this abstraction, not concrete implementations.
"""

# removed - breaks FastAPI

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clarity.core.config_aws import MiddlewareConfig
else:
    # For runtime, we need the actual import
    try:
        from clarity.core.config_aws import MiddlewareConfig
    except ImportError:
        MiddlewareConfig = Any


class IConfigProvider(ABC):
    """Interface for configuration management.

    Provides access to application configuration following
    Dependency Inversion Principle.
    """

    @abstractmethod
    def get_setting(
        self, key: str, *, default: str | int | bool | None = None
    ) -> str | int | bool | None:
        """Get configuration setting.

        Args:
            key: Configuration key to retrieve
            default: Default value if key not found

        Returns:
            Configuration value or default
        """

    @abstractmethod
    def is_development(self) -> bool:
        """Check if running in development mode.

        Returns:
            True if in development mode, False otherwise
        """

    @abstractmethod
    def should_skip_external_services(self) -> bool:
        """Check if external services should be skipped during startup.

        Returns:
            True if external services should be skipped, False otherwise
        """

    @abstractmethod
    def is_auth_enabled(self) -> bool:
        """Check if authentication is enabled.

        Returns:
            True if authentication is enabled, False otherwise
        """

    @abstractmethod
    def get_aws_region(self) -> str:
        """Get AWS region.

        Returns:
            AWS region
        """

    @abstractmethod
    def get_log_level(self) -> str:
        """Get logging level.

        Returns:
            Logging level string
        """

    @abstractmethod
    def get_middleware_config(self) -> MiddlewareConfig:
        """Get environment-specific middleware configuration.

        Returns:
            Middleware configuration object
        """

    @abstractmethod
    def get_settings_model(self) -> Any:
        """Get the settings model instance.

        Returns:
            Settings model instance
        """

    def get_dynamodb_url(self) -> str:
        """Get DynamoDB service URL."""
        raise NotImplementedError
