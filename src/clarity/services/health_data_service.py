"""CLARITY Digital Twin Platform - Health Data Service.

Business logic layer for health data processing, validation, and management.
Provides enterprise-grade health data handling with HIPAA compliance features.

Following Robert C. Martin's Clean Architecture principles.
"""

from datetime import UTC, datetime
import logging
from typing import Any
import uuid

from clarity.core.interfaces import IHealthDataRepository
from clarity.models.health_data import (
    HealthDataResponse,
    HealthDataUpload,
    HealthMetric,
    ProcessingStatus,
)

# Configure logger
logger = logging.getLogger(__name__)


# Custom exceptions
class HealthDataServiceError(Exception):
    """Base exception for health data service operations."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class DataNotFoundError(HealthDataServiceError):
    """Exception raised when requested data is not found."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=404)


def _raise_validation_error(error_summary: str) -> None:
    """Raise validation error exception."""
    msg = f"Health data validation failed: {error_summary}"
    raise HealthDataServiceError(msg, status_code=400)


def _raise_data_not_found_error(processing_id: str) -> None:
    """Raise data not found error exception."""
    msg = f"Processing job {processing_id} not found"
    raise DataNotFoundError(msg)


class HealthDataService:
    """Service layer for health data processing and management.

    Handles business logic for:
    - Health data validation and processing
    - Repository storage operations via abstraction
    - Processing status tracking
    - Business rule enforcement
    - Audit trail maintenance

    Follows Clean Architecture by depending on IHealthDataRepository interface.
    """

    def __init__(self, repository: IHealthDataRepository) -> None:
        """Initialize health data service.

        Args:
            repository: Health data repository implementing IHealthDataRepository
        """
        self.repository = repository
        self.logger = logging.getLogger(__name__)

    async def process_health_data(
        self, health_data: HealthDataUpload
    ) -> HealthDataResponse:
        """Process and validate health data upload.

        Args:
            health_data: Health data upload containing metrics and metadata

        Returns:
            Processing response with job ID and initial status

        Raises:
            HealthDataServiceError: If processing fails
        """
        try:
            self.logger.info("Processing health data for user: %s", health_data.user_id)

            # Generate unique processing ID
            processing_id = str(uuid.uuid4())

            # Validate health metrics
            validation_errors: list[str] = []
            for metric in health_data.metrics:
                try:
                    # Basic validation - check if metric_type and created_at exist
                    if not metric.metric_type or not metric.created_at:
                        validation_errors.append(
                            f"Metric {metric.metric_id} missing required fields"
                        )
                    elif not self._validate_metric_business_rules(metric):
                        validation_errors.append(
                            f"Metric {metric.metric_id} failed business validation"
                        )
                except ValueError as e:
                    validation_errors.append(f"Metric {metric.metric_id}: {e!s}")

            # Check if validation passed
            if validation_errors:
                error_summary = f"Validation failed: {len(validation_errors)} errors"
                self.logger.warning("%s", error_summary)
                _raise_validation_error(error_summary)

            # Store health data using repository
            await self.repository.save_health_data(
                user_id=str(health_data.user_id),
                processing_id=processing_id,
                metrics=health_data.metrics,
                upload_source=health_data.upload_source,
                client_timestamp=health_data.client_timestamp,
            )

            # Log operation
            self.logger.info("Health data processing initiated: %s", processing_id)

            return HealthDataResponse(
                processing_id=uuid.UUID(processing_id),
                status=ProcessingStatus.PROCESSING,
                accepted_metrics=len(health_data.metrics),
                rejected_metrics=0,
                validation_errors=[],  # Return empty list for now
                estimated_processing_time=len(health_data.metrics)
                * 2,  # 2 seconds per metric
                sync_token=health_data.sync_token,
                message="Health data processing initiated successfully",
                timestamp=datetime.now(UTC),
            )

        except HealthDataServiceError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            self.logger.exception("Unexpected error during health data processing")
            msg = f"Health data processing failed: {e!s}"
            raise HealthDataServiceError(msg) from e

    async def get_processing_status(
        self, processing_id: str, user_id: str
    ) -> dict[str, Any] | None:
        """Get processing status for a health data upload.

        Returns status information or None if not found.
        Implements user-scoped data access control.
        """
        try:
            self.logger.debug(
                "Getting processing status: %s for user: %s", processing_id, user_id
            )

            # Get status from repository with user validation
            status_info = await self.repository.get_processing_status(
                processing_id=processing_id, user_id=user_id
            )

            if not status_info:
                _raise_data_not_found_error(processing_id)

        except Exception as e:
            if isinstance(e, (DataNotFoundError, HealthDataServiceError)):
                raise
            self.logger.exception("Error getting processing status")
            msg = f"Failed to get processing status: {e!s}"
            raise HealthDataServiceError(msg) from e
        else:
            return status_info

    async def get_user_health_data(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        metric_type: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Retrieve user's health data with filtering and pagination.

        Args:
            user_id: User ID to retrieve data for
            limit: Maximum number of records to return
            offset: Number of records to skip
            metric_type: Filter by specific metric type
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            User health data with metadata

        Raises:
            HealthDataServiceError: If retrieval operation fails
        """
        try:
            self.logger.debug("Retrieving health data for user: %s", user_id)

            # Get user health data from repository
            health_data = await self.repository.get_user_health_data(
                user_id=user_id,
                limit=limit,
                offset=offset,
                metric_type=metric_type,
                start_date=start_date,
                end_date=end_date,
            )

            self.logger.info(
                "Retrieved %s health records for user: %s",
                len(health_data.get("metrics", [])),  # type: ignore[arg-type]
                user_id,
            )

        except Exception as e:
            self.logger.exception("Error retrieving health data")
            msg = f"Failed to retrieve health data: {e!s}"
            raise HealthDataServiceError(msg) from e
        else:
            return health_data

    async def delete_health_data(
        self, user_id: str, processing_id: str | None = None
    ) -> bool:
        """Delete user's health data with audit trail.

        Args:
            user_id: User ID to delete data for
            processing_id: Optional specific processing job to delete

        Returns:
            True if deletion was successful

        Raises:
            HealthDataServiceError: If deletion operation fails
        """
        try:
            self.logger.info(
                "Deleting health data: %s for user: %s", processing_id, user_id
            )

            # Delete health data using repository
            success = await self.repository.delete_health_data(
                user_id=user_id, processing_id=processing_id
            )

            if success:
                self.logger.info(
                    "User health data deleted",
                    extra={
                        "user_id": user_id,
                        "processing_id": processing_id,
                    },
                )

        except Exception as e:
            self.logger.exception("Error during data deletion")
            msg = f"Failed to delete health data: {e!s}"
            raise HealthDataServiceError(msg) from e
        else:
            return success

    @staticmethod
    def _validate_metric_business_rules(metric: HealthMetric) -> bool:
        """Validate health metric against business rules."""
        try:
            # Check for required metric type
            if not metric.metric_type:
                return False

            # Validate biometric data ranges
            if (
                metric.metric_type.value in {"heart_rate", "blood_pressure"}
                and metric.biometric_data
            ):
                return True

            # Validate sleep data
            if metric.metric_type.value == "sleep_analysis" and metric.sleep_data:
                return True

            # Validate activity data
            if metric.metric_type.value == "activity_level" and metric.activity_data:
                return True

            # Validate mental health data
            return bool(
                metric.metric_type.value == "mood_assessment"
                and metric.mental_health_data is not None
            )

        except (ValueError, AttributeError, TypeError) as e:
            logger.warning("Business rule validation failed: %s", e)
            return False
