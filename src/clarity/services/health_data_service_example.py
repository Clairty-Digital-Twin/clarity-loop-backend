"""ARCHITECTURAL EXAMPLE: Enhanced Health Data Service.

⚠️  This file demonstrates the new architectural patterns and improvements.
    For production use, see: src/clarity/services/health_data_service.py

📚 DEMONSTRATES:
- ✅ New decorator patterns for cross-cutting concerns
- ✅ Ports layer interfaces instead of core.interfaces
- ✅ Model integrity verification
- ✅ Comprehensive audit trail and enhanced logging
- ✅ Enhanced error handling and business rule validation

💡 PURPOSE:
This serves as a reference implementation showing how to upgrade existing
services with the new architectural patterns. Copy patterns from here into
production services as needed.

🔄 MIGRATION:
When ready to use these patterns in production:
1. Copy decorator usage patterns to health_data_service.py
2. Update imports to use ports layer
3. Add model integrity checks
4. Enhance error handling as shown here
5. Remove this example file
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

# Import from new ports layer instead of core interfaces
from clarity.core.decorators import (
    audit_trail,
    log_execution,
    measure_execution_time,
    retry_on_failure,
    service_method,
)
from clarity.ml.model_integrity import verify_startup_models
from clarity.models.health_data import (
    HealthDataResponse,
    HealthDataUpload,
    HealthMetric,
    ProcessingStatus,
)
from clarity.ports.data_ports import IHealthDataRepository

# Configure logger
logger = logging.getLogger(__name__)

# Constants
MAX_QUERY_LIMIT = 1000
MIN_QUERY_LIMIT = 1
MIN_QUERY_OFFSET = 0


# Custom exceptions for better error handling
class HealthDataServiceError(Exception):
    """Service-specific error for health data operations."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class DataNotFoundError(Exception):
    """Raised when requested data is not found."""


class ValidationError(Exception):
    """Raised when data validation fails."""


class EnhancedHealthDataService:
    """Enhanced Health Data Service with architectural improvements.

    Demonstrates the new architectural patterns including:
    - Decorator patterns for cross-cutting concerns
    - Ports layer interfaces for clean dependency injection
    - Model integrity verification before ML operations
    - Comprehensive audit trails and enhanced logging
    - Sophisticated error handling and business rule validation

    This serves as a reference implementation for upgrading existing services.
    """

    def __init__(self, repository: IHealthDataRepository):
        """Initialize the enhanced health data service.

        Args:
            repository: Health data repository implementation
        """
        self.repository = repository
        self.logger = logger

    @service_method(log_level=logging.INFO, timing_threshold_ms=500.0)
    @audit_trail("process_health_data", user_id_param="user_id")
    async def process_health_data(
        self, health_data: HealthDataUpload, user_id: str
    ) -> HealthDataResponse:
        """Process and validate health data upload.

        Enhanced version with:
        - Model integrity verification before processing
        - Detailed validation with specific error reporting
        - Comprehensive audit trail
        - Enhanced error handling

        Args:
            health_data: Health data to process
            user_id: ID of the user submitting the data

        Returns:
            Processing response with status and metadata

        Raises:
            HealthDataServiceError: If processing fails
        """
        try:
            # Verify ML models are intact before processing
            if not await self._verify_processing_models():
                self._raise_model_integrity_error()

            # Generate unique processing ID
            processing_id = str(uuid.uuid4())

            # Enhanced validation with detailed error reporting
            validation_errors = await self._validate_health_metrics(health_data.metrics)

            if validation_errors:
                error_summary = f"Validation failed: {len(validation_errors)} errors"
                self.logger.warning(
                    "Health data validation failed: %s", validation_errors
                )
                raise HealthDataServiceError(error_summary, status_code=400)

            # Store health data using repository with enhanced error handling
            success = await self.repository.save_health_data(
                user_id=user_id,
                processing_id=processing_id,
                metrics=health_data.metrics,
                upload_source=health_data.upload_source,
                client_timestamp=health_data.client_timestamp,
            )

            if not success:
                self._raise_storage_error()

            return HealthDataResponse(
                processing_id=processing_id,
                status=ProcessingStatus.PROCESSING,
                accepted_metrics=len(health_data.metrics),
                rejected_metrics=0,
                validation_errors=[],
                estimated_processing_time=len(health_data.metrics) * 2,
                sync_token=health_data.sync_token,
                message="Health data received and queued for processing",
                timestamp=datetime.now(UTC),
            )

        except (HealthDataServiceError, ValidationError):
            # Re-raise our custom exceptions as-is
            raise
        except Exception as e:
            self.logger.exception("Unexpected error during health data processing")
            error_msg = f"Health data processing failed: {e}"
            raise HealthDataServiceError(error_msg) from e

    @log_execution(level=logging.DEBUG)
    @measure_execution_time(threshold_ms=100.0)
    async def get_processing_status(
        self, processing_id: str, user_id: str
    ) -> dict[str, Any]:
        """Get the current processing status for a health data upload.

        Enhanced with:
        - Detailed error logging
        - Performance monitoring
        - Enhanced error messages

        Args:
            processing_id: ID of the processing job
            user_id: ID of the user (for authorization)

        Returns:
            Processing status information

        Raises:
            HealthDataServiceError: If status retrieval fails
            DataNotFoundError: If processing job not found
        """
        try:
            # Get status information from repository
            status_info = await self.repository.get_processing_status(
                processing_id, user_id
            )

            if not status_info:
                error_msg = f"Processing job {processing_id} not found"
                raise DataNotFoundError(error_msg)

            return status_info

        except (DataNotFoundError, HealthDataServiceError):
            # Re-raise our specific exceptions
            raise
        except Exception as e:
            self.logger.exception("Error getting processing status")
            error_msg = f"Failed to get processing status: {e}"
            raise HealthDataServiceError(error_msg) from e

    @service_method(log_level=logging.INFO, timing_threshold_ms=200.0)
    @retry_on_failure(max_retries=2, exponential_backoff=True)
    async def get_user_health_data(
        self,
        user_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Retrieve health data for a user with enhanced validation.

        Enhanced with:
        - Parameter validation with specific error messages
        - Retry mechanism for transient failures
        - Performance monitoring

        Args:
            user_id: ID of the user
            start_date: Start date for data range (optional)
            end_date: End date for data range (optional)
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of health data records

        Raises:
            HealthDataServiceError: If retrieval fails or parameters invalid
        """
        try:
            # Validate parameters
            if limit <= MIN_QUERY_LIMIT or limit > MAX_QUERY_LIMIT:
                self._raise_invalid_limit_error()

            if offset < MIN_QUERY_OFFSET:
                self._raise_invalid_offset_error()

            return await self.repository.get_user_health_data(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                offset=offset,
            )

        except HealthDataServiceError:
            # Re-raise our specific exceptions
            raise
        except Exception as e:
            self.logger.exception("Error retrieving health data")
            error_msg = f"Failed to retrieve health data: {e}"
            raise HealthDataServiceError(error_msg) from e

    @audit_trail(
        "delete_health_data", user_id_param="user_id", resource_id_param="processing_id"
    )
    @service_method(log_level=logging.WARNING, timing_threshold_ms=1000.0)
    async def delete_health_data(
        self, user_id: str, processing_id: str | None = None
    ) -> bool:
        """Delete health data with enhanced business rule validation.

        Enhanced with:
        - Business rule validation (retention policies)
        - Comprehensive audit trail
        - Enhanced authorization checks

        Args:
            user_id: ID of the user
            processing_id: Specific processing ID to delete (optional)

        Returns:
            True if deletion successful

        Raises:
            HealthDataServiceError: If deletion fails or not allowed
        """
        try:
            # Check data retention policies (example business rule)
            if not await self._can_delete_data(user_id, processing_id):
                self._raise_retention_policy_error()

            success = await self.repository.delete_health_data(
                user_id=user_id, processing_id=processing_id
            )

            if not success:
                self._raise_deletion_failed_error()

            return success

        except HealthDataServiceError:
            # Re-raise our specific exceptions
            raise
        except Exception as e:
            self.logger.exception("Error during data deletion")
            error_msg = f"Failed to delete health data: {e}"
            raise HealthDataServiceError(error_msg) from e

    # Private helper methods with decorators

    @log_execution(level=logging.DEBUG)
    async def _verify_processing_models(self) -> bool:
        """Verify that ML models are intact before processing."""
        try:
            return verify_startup_models()
        except Exception as e:
            self.logger.exception("Model integrity verification failed: %s", e)
            return False

    @log_execution(level=logging.DEBUG)
    async def _validate_health_metrics(self, metrics: list[HealthMetric]) -> list[str]:
        """Enhanced validation with detailed error reporting."""
        validation_errors: list[str] = []

        for metric in metrics:
            try:
                # Basic data validation
                if not self._validate_metric_business_rules(metric):
                    validation_errors.append(
                        f"Metric {metric.metric_id}: failed business rule validation"
                    )

            except ValueError as e:
                validation_errors.append(f"Metric {metric.metric_id}: {e}")

        return validation_errors

    @log_execution(level=logging.DEBUG)
    async def _can_delete_data(self, user_id: str, processing_id: str | None) -> bool:  # noqa: ARG002
        """Check if data can be deleted according to business rules."""
        # Example business rule: Check if data is within retention period
        try:
            # In a real implementation, you would check:
            # - Data retention periods
            # - User consent status
            # - legal holds, consent status, etc.
            return True
        except Exception as e:
            self.logger.warning("Error checking deletion policy: %s", e)
            return False

    @staticmethod
    def _validate_metric_business_rules(metric: HealthMetric) -> bool:
        """Validate health metric against business rules."""
        try:
            # Example business rules validation
            if not metric.value or metric.value <= 0:
                return False

            # Type-specific validation
            metric_type_value = metric.metric_type.value.lower()

            if metric_type_value in {"heart_rate", "blood_pressure"}:
                return bool(metric.biometric_data)
            if metric_type_value == "sleep_analysis":
                return bool(metric.sleep_data)
            if metric_type_value == "activity_level":
                return bool(metric.activity_data)
            if metric_type_value == "mental_health":
                return metric.mental_health_data is not None
            # Unknown metric type
            return False

        except (ValueError, AttributeError, TypeError) as e:
            logger.warning("Error validating metric business rules: %s", e)
            return False

    # Error raising helper methods to satisfy TRY301

    def _raise_model_integrity_error(self) -> None:
        """Raise model integrity verification error."""
        msg = "ML model integrity verification failed. Processing cannot continue."
        raise HealthDataServiceError(msg, status_code=503)

    def _raise_storage_error(self) -> None:
        """Raise storage error."""
        msg = "Failed to store health data in repository"
        raise HealthDataServiceError(msg, status_code=500)

    def _raise_invalid_limit_error(self) -> None:
        """Raise invalid limit parameter error."""
        msg = "Invalid limit parameter"
        raise HealthDataServiceError(msg, status_code=400)

    def _raise_invalid_offset_error(self) -> None:
        """Raise invalid offset parameter error."""
        msg = "Invalid offset parameter"
        raise HealthDataServiceError(msg, status_code=400)

    def _raise_retention_policy_error(self) -> None:
        """Raise retention policy error."""
        msg = "Data deletion not allowed due to retention policy"
        raise HealthDataServiceError(msg, status_code=403)

    def _raise_deletion_failed_error(self) -> None:
        """Raise deletion failed error."""
        msg = "Data deletion failed at repository level"
        raise HealthDataServiceError(msg, status_code=500)


def create_enhanced_health_data_service(
    repository: IHealthDataRepository,
) -> EnhancedHealthDataService:
    """Factory function to create enhanced health data service.

    Demonstrates proper dependency injection using the ports layer.

    Args:
        repository: Health data repository implementation

    Returns:
        Configured enhanced health data service
    """
    return EnhancedHealthDataService(repository)
