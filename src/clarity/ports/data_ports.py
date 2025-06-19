"""Data port interfaces.

Defines the contract for data repositories following Clean Architecture.
Business logic layer depends on this abstraction, not concrete implementations.
"""

# removed - breaks FastAPI

from abc import ABC, abstractmethod
from datetime import datetime

from clarity.models.health_data import HealthMetric


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
        """Save health data for a user (legacy method).

        Args:
            user_id: User identifier
            data: Health data to save

        Returns:
            Record identifier
        """

    @abstractmethod
    async def get_data(
        self, user_id: str, filters: dict[str, str] | None = None
    ) -> dict[str, str]:
        """Retrieve health data for a user (legacy method).

        Args:
            user_id: User identifier
            filters: Optional filters to apply

        Returns:
            Health data dictionary
        """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the repository.

        Performs any necessary setup operations like connecting to database,
        creating indexes, etc.
        """

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up repository resources.

        Performs cleanup operations like closing connections, releasing resources, etc.
        """
