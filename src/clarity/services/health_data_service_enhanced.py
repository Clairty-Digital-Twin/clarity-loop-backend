"""ARCHITECTURAL EXAMPLE: Enhanced Health Data Service

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

from datetime import UTC, datetime
import logging
from typing import Any
import uuid

# Import from new ports layer instead of core interfaces
from clarity.ports.data_ports import IHealthDataRepository
from clarity.core.decorators import (
    audit_trail,
    log_execution,
    measure_execution_time,
    service_method,
    retry_on_failure,
)
from clarity.ml.model_integrity import verify_startup_models
from clarity.models.health_data import (
    HealthDataResponse,
    HealthDataUpload,
    HealthMetric,
    ProcessingStatus,
)

# Configure logger
logger = logging.getLogger(__name__)


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


class EnhancedHealthDataService:
    """Enhanced Health Data Service with architectural improvements.
    
    Demonstrates the new architectural patterns including:
    - Decorator patterns for cross-cutting concerns
    - Ports layer for clean interfaces
    - Model integrity verification
    - Enhanced audit trails and monitoring
    """

    def __init__(self, repository: IHealthDataRepository) -> None:
        """Initialize enhanced health data service.

        Args:
            repository: Health data repository implementing IHealthDataRepository
        """
        self.repository = repository
        self.logger = logging.getLogger(__name__)

    @service_method(log_level=logging.INFO, timing_threshold_ms=500.0)
    @audit_trail("process_health_data", user_id_param="user_id")
    async def process_health_data(
        self, health_data: HealthDataUpload, user_id: str
    ) -> HealthDataResponse:
        """Process and validate health data upload.

        Enhanced with:
        - Automatic logging and timing
        - Audit trail generation
        - Model integrity verification
        - Business rule validation

        Args:
            health_data: Health data upload containing metrics and metadata
            user_id: User ID for audit trail (extracted from health_data)

        Returns:
            Processing response with job ID and initial status

        Raises:
            HealthDataServiceError: If processing fails
        """
        try:
            # Verify ML models are intact before processing
            if not await self._verify_processing_models():
                raise HealthDataServiceError(
                    "ML model integrity verification failed. Processing cannot continue.",
                    status_code=503
                )

            # Generate unique processing ID
            processing_id = str(uuid.uuid4())

            # Enhanced validation with detailed error reporting
            validation_errors = await self._validate_health_metrics(health_data.metrics)
            
            if validation_errors:
                error_summary = f"Validation failed: {len(validation_errors)} errors"
                self.logger.warning("Health data validation failed: %s", validation_errors)
                raise HealthDataServiceError(error_summary, status_code=400)

            # Store health data using repository with enhanced error handling
            success = await self.repository.save_health_data(
                user_id=str(health_data.user_id),
                processing_id=processing_id,
                metrics=health_data.metrics,
                upload_source=health_data.upload_source,
                client_timestamp=health_data.client_timestamp,
            )

            if not success:
                raise HealthDataServiceError(
                    "Failed to store health data in repository",
                    status_code=500
                )

            return HealthDataResponse(
                processing_id=uuid.UUID(processing_id),
                status=ProcessingStatus.PROCESSING,
                accepted_metrics=len(health_data.metrics),
                rejected_metrics=0,
                validation_errors=[],
                estimated_processing_time=len(health_data.metrics) * 2,
                sync_token=health_data.sync_token,
                message="Health data processing initiated successfully",
                timestamp=datetime.now(UTC),
            )

        except HealthDataServiceError:
            raise
        except Exception as e:
            self.logger.exception("Unexpected error during health data processing")
            raise HealthDataServiceError(f"Health data processing failed: {e}") from e

    @log_execution(level=logging.DEBUG)
    @measure_execution_time(threshold_ms=100.0)
    @retry_on_failure(max_retries=2, delay_seconds=0.5)
    async def get_processing_status(
        self, processing_id: str, user_id: str
    ) -> dict[str, Any]:
        """Get processing status with enhanced reliability.

        Enhanced with:
        - Automatic retry on transient failures
        - Performance monitoring
        - Detailed logging

        Args:
            processing_id: Processing job identifier
            user_id: User identifier for access control

        Returns:
            Processing status information

        Raises:
            DataNotFoundError: If processing job not found
            HealthDataServiceError: If retrieval fails
        """
        try:
            status_info = await self.repository.get_processing_status(
                processing_id=processing_id, user_id=user_id
            )

            if not status_info:
                raise DataNotFoundError(f"Processing job {processing_id} not found")

            return status_info

        except (DataNotFoundError, HealthDataServiceError):
            raise
        except Exception as e:
            self.logger.exception("Error getting processing status")
            raise HealthDataServiceError(f"Failed to get processing status: {e}") from e

    @service_method(log_level=logging.INFO, timing_threshold_ms=200.0)
    @audit_trail("get_user_health_data", user_id_param="user_id")
    async def get_user_health_data(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        metric_type: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Retrieve user's health data with enhanced monitoring.

        Enhanced with:
        - Audit trail for data access
        - Performance monitoring
        - Parameter validation

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
            # Validate parameters
            if limit <= 0 or limit > 1000:
                raise HealthDataServiceError("Invalid limit parameter", status_code=400)
            
            if offset < 0:
                raise HealthDataServiceError("Invalid offset parameter", status_code=400)

            health_data = await self.repository.get_user_health_data(
                user_id=user_id,
                limit=limit,
                offset=offset,
                metric_type=metric_type,
                start_date=start_date,
                end_date=end_date,
            )

            return health_data

        except HealthDataServiceError:
            raise
        except Exception as e:
            self.logger.exception("Error retrieving health data")
            raise HealthDataServiceError(f"Failed to retrieve health data: {e}") from e

    @audit_trail("delete_health_data", user_id_param="user_id", resource_id_param="processing_id")
    @service_method(log_level=logging.WARNING, timing_threshold_ms=300.0)
    async def delete_health_data(
        self, user_id: str, processing_id: str | None = None
    ) -> bool:
        """Delete user's health data with comprehensive audit trail.

        Enhanced with:
        - Comprehensive audit logging
        - Performance monitoring
        - Data retention policy checks

        Args:
            user_id: User ID to delete data for
            processing_id: Optional specific processing job to delete

        Returns:
            True if deletion was successful

        Raises:
            HealthDataServiceError: If deletion operation fails
        """
        try:
            # Check data retention policies (example business rule)
            if not await self._can_delete_data(user_id, processing_id):
                raise HealthDataServiceError(
                    "Data deletion not allowed due to retention policy",
                    status_code=403
                )

            success = await self.repository.delete_health_data(
                user_id=user_id, processing_id=processing_id
            )

            if not success:
                raise HealthDataServiceError(
                    "Data deletion failed at repository level",
                    status_code=500
                )

            return success

        except HealthDataServiceError:
            raise
        except Exception as e:
            self.logger.exception("Error during data deletion")
            raise HealthDataServiceError(f"Failed to delete health data: {e}") from e

    # Private helper methods with decorators

    @log_execution(level=logging.DEBUG)
    async def _verify_processing_models(self) -> bool:
        """Verify ML model integrity before processing."""
        try:
            return verify_startup_models()
        except Exception as e:
            self.logger.error("Model integrity verification failed: %s", e)
            return False

    @measure_execution_time(threshold_ms=50.0)
    async def _validate_health_metrics(self, metrics: list[HealthMetric]) -> list[str]:
        """Enhanced validation with detailed error reporting."""
        validation_errors: list[str] = []
        
        for metric in metrics:
            try:
                if not metric.metric_type or not metric.created_at:
                    validation_errors.append(
                        f"Metric {metric.metric_id} missing required fields"
                    )
                elif not self._validate_metric_business_rules(metric):
                    validation_errors.append(
                        f"Metric {metric.metric_id} failed business validation"
                    )
            except ValueError as e:
                validation_errors.append(f"Metric {metric.metric_id}: {e}")
        
        return validation_errors

    @log_execution(level=logging.DEBUG)
    async def _can_delete_data(self, user_id: str, processing_id: str | None) -> bool:
        """Check if data can be deleted according to business rules."""
        # Example business rule: Check if data is within retention period
        try:
            # In a real implementation, this would check retention policies,
            # legal holds, consent status, etc.
            return True
        except Exception as e:
            self.logger.warning("Error checking deletion policy: %s", e)
            return False

    @staticmethod
    def _validate_metric_business_rules(metric: HealthMetric) -> bool:
        """Validate health metric against business rules."""
        try:
            if not metric.metric_type:
                return False

            # Enhanced validation with specific checks per metric type
            metric_type_value = metric.metric_type.value

            if metric_type_value in {"heart_rate", "blood_pressure"}:
                return bool(metric.biometric_data)
            elif metric_type_value == "sleep_analysis":
                return bool(metric.sleep_data)
            elif metric_type_value == "activity_level":
                return bool(metric.activity_data)
            elif metric_type_value == "mood_assessment":
                return metric.mental_health_data is not None
            else:
                # Unknown metric type
                return False

        except (ValueError, AttributeError, TypeError) as e:
            logger.warning("Business rule validation failed: %s", e)
            return False


# Example factory function demonstrating dependency injection with ports
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