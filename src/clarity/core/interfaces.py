"""Core interfaces for dependency inversion.

Following Uncle Bob's Clean Architecture principles, this module defines
abstract interfaces that allow higher-level modules to not depend on
lower-level modules, but both depend on abstractions.
"""

from abc import ABC, abstractmethod
from typing import Any

from fastapi import Request


class IAuthProvider(ABC):
    """Abstract authentication provider interface."""

    @abstractmethod
    async def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify an authentication token."""

    @abstractmethod
    async def get_user_info(self, user_id: str) -> dict[str, Any] | None:
        """Get user information by ID."""


class IMiddleware(ABC):
    """Abstract middleware interface."""

    @abstractmethod
    async def __call__(self, request: Request, call_next: Any) -> Any:
        """Process request through middleware."""


class IHealthDataRepository(ABC):
    """Abstract repository for health data operations."""

    @abstractmethod
    async def save_data(self, user_id: str, data: dict[str, Any]) -> str:
        """Save health data for a user."""

    @abstractmethod
    async def get_data(
        self, user_id: str, filters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Retrieve health data for a user."""


class IConfigProvider(ABC):
    """Abstract configuration provider interface."""

    @abstractmethod
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get configuration setting."""

    @abstractmethod
    def is_development(self) -> bool:
        """Check if running in development mode."""

    @abstractmethod
    def is_auth_enabled(self) -> bool:
        """Check if authentication is enabled."""

    @abstractmethod
    def get_firebase_config(self) -> dict[str, Any]:
        """Get Firebase configuration."""

    @abstractmethod
    def get_gcp_project_id(self) -> str:
        """Get Google Cloud Platform project ID."""

    @abstractmethod
    def get_log_level(self) -> str:
        """Get logging level."""
