"""Core interfaces for dependency inversion.

Following Uncle Bob's Clean Architecture principles, this module defines
abstract interfaces that allow higher-level modules to not depend on
lower-level modules, but both depend on abstractions.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from fastapi import Request, Response

from clarity.models.health_data import HealthMetric


class IAuthProvider(ABC):
    """Abstract authentication provider interface."""

    @abstractmethod
    async def verify_token(self, token: str) -> dict[str, str] | None:
        """Verify an authentication token."""

    @abstractmethod
    async def get_user_info(self, user_id: str) -> dict[str, str] | None:
        """Get user information by ID."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the authentication provider."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up authentication provider resources."""


class IMiddleware(ABC):
    """Interface for HTTP middleware components.

    Following Clean Architecture:
    - Middleware operates at the interface adapter layer
    - Handles cross-cutting concerns (auth, logging, etc.)
    - Should not contain business logic
    """

    @abstractmethod
    async def __call__(
        self, request: "Request", call_next: "Awaitable[Response]"
    ) -> "Response":
        """Process request through middleware."""


class IHealthDataRepository(ABC):
    """Abstract repository for health data operations.

    Defines the contract for health data persistence according to Clean Architecture.
    Business logic layer depends on this abstraction, not concrete implementations.
    """

    @abstractmethod
    async def save_health_data(
        self,
        user_id: str,
        processing_id: str,
        metrics: list[HealthMetric],
        upload_source: str,
        client_timestamp: datetime,
    ) -> bool:
        """Save health data with processing metadata.

        Args:
            user_id: User identifier
            processing_id: Processing job identifier
            metrics: List of health metrics
            upload_source: Source of the upload
            client_timestamp: Client-side timestamp

        Returns:
            True if saved successfully
        """

    @abstractmethod
    async def get_user_health_data(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        metric_type: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, str]:
        """Retrieve user health data with filtering and pagination.

        Args:
            user_id: User identifier
            limit: Maximum records to return
            offset: Records to skip
            metric_type: Filter by metric type
            start_date: Filter from date
            end_date: Filter to date

        Returns:
            Health data with pagination metadata
        """

    @abstractmethod
    async def get_processing_status(
        self, processing_id: str, user_id: str
    ) -> dict[str, str] | None:
        """Get processing status for a health data upload.

        Args:
            processing_id: Processing job identifier
            user_id: User identifier for ownership verification

        Returns:
            Processing status info or None if not found
        """

    @abstractmethod
    async def delete_health_data(
        self, user_id: str, processing_id: str | None = None
    ) -> bool:
        """Delete user health data.

        Args:
            user_id: User identifier
            processing_id: Optional specific processing job to delete

        Returns:
            True if deletion was successful
        """

    @abstractmethod
    async def save_data(self, user_id: str, data: dict[str, str]) -> str:
        """Save health data for a user (legacy method)."""

    @abstractmethod
    async def get_data(
        self, user_id: str, filters: dict[str, str] | None = None
    ) -> dict[str, str]:
        """Retrieve health data for a user (legacy method)."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the repository."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up repository resources."""


class IConfigProvider(ABC):
    """Interface for configuration management.

    Provides access to application configuration following
    Dependency Inversion Principle.
    """

    @abstractmethod
    def get_setting(
        self, key: str, default: str | int | bool | None = None
    ) -> str | int | bool | None:
        """Get configuration setting."""

    @abstractmethod
    def is_development(self) -> bool:
        """Check if running in development mode."""

    @abstractmethod
    def should_skip_external_services(self) -> bool:
        """Check if external services should be skipped during startup."""

    @abstractmethod
    def is_auth_enabled(self) -> bool:
        """Check if authentication is enabled."""

    @abstractmethod
    def get_firebase_config(self) -> dict[str, str]:
        """Get Firebase configuration."""

    @abstractmethod
    def get_gcp_project_id(self) -> str:
        """Get Google Cloud Platform project ID."""

    @abstractmethod
    def get_log_level(self) -> str:
        """Get logging level."""
