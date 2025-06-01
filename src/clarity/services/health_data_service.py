"""CLARITY Digital Twin Platform - Health Data Service.

Business logic layer for health data processing, validation, and management.
Provides enterprise-grade health data handling with HIPAA compliance features.
"""

from datetime import UTC, datetime
import logging
from typing import Any
import uuid

from clarity.models.health_data import (
    HealthDataResponse,
    HealthDataUpload,
    HealthMetric,
    ProcessingStatus,
)
from clarity.storage.firestore_client import (
    DocumentNotFoundError,
    FirestoreClient,
    FirestoreError,
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


class HealthDataService:
    """Service layer for health data processing and management.

    Handles business logic for:
    - Health data validation and processing
    - Firestore storage operations
    - Processing status tracking
    - Business rule enforcement
    - Audit trail maintenance
    """

    def __init__(self, firestore_client: FirestoreClient) -> None:
        """Initialize health data service.

        Args:
            firestore_client: Configured Firestore client for data persistence
        """
        self.firestore_client = firestore_client
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
                msg = f"Health data validation failed: {error_summary}"
                raise HealthDataServiceError(msg, status_code=400)

            # Create processing document
            processing_doc = {
                "processing_id": processing_id,
                "user_id": str(health_data.user_id),  # Convert UUID to string
                "upload_source": health_data.upload_source,
                "client_timestamp": health_data.client_timestamp.isoformat(),
                "created_at": datetime.now(UTC).isoformat(),
                "status": ProcessingStatus.PROCESSING.value,
                "total_metrics": len(health_data.metrics),
                "processed_metrics": 0,
                "validation_errors": validation_errors,
            }

            # Store processing document
            await self.firestore_client.create_document(
                collection="health_data_processing", data=processing_doc
            )

            # Store metrics asynchronously
            await self._store_metrics(
                str(health_data.user_id), health_data.metrics, processing_id
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
    ) -> dict[str, Any]:
        """Get processing status for a health data upload job.

        Args:
            processing_id: Unique identifier for the processing job
            user_id: User ID to verify ownership

        Returns:
            Processing status information

        Raises:
            DataNotFoundError: If processing job not found
            HealthDataServiceError: If retrieval operation fails
        """
        try:
            # Retrieve processing record
            doc = await self.firestore_client.get_document(
                collection="processing_jobs", document_id=processing_id
            )

            if not doc:
                raise DataNotFoundError(f"Processing job {processing_id} not found")

            # Verify user ownership
            if doc.get("user_id") != user_id:
                raise DataNotFoundError(f"Processing job {processing_id} not found")

            return {
                "processing_id": processing_id,
                "status": doc.get("status"),
                "progress": self._calculate_progress(doc),
                "estimated_completion": doc.get("estimated_completion"),
                "accepted_metrics": doc.get("accepted_metrics", 0),
                "rejected_metrics": doc.get("rejected_metrics", 0),
                "validation_errors": doc.get("validation_errors", []),
                "created_at": doc.get("server_timestamp"),
                "completed_at": doc.get("completed_at"),
            }

        except DocumentNotFoundError:
            raise DataNotFoundError(f"Processing job {processing_id} not found")
        except FirestoreError as e:
            self.logger.exception("Firestore error retrieving processing status: %s", e)
            raise HealthDataServiceError(f"Failed to retrieve processing status: {e!s}")

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
            # Build query filters
            filters = [{"field": "user_id", "op": "==", "value": user_id}]

            if metric_type:
                filters.append(
                    {"field": "metric_type", "op": "==", "value": metric_type}
                )

            if start_date:
                filters.append(
                    {"field": "created_at", "op": ">=", "value": start_date.isoformat()}
                )

            if end_date:
                filters.append(
                    {"field": "created_at", "op": "<=", "value": end_date.isoformat()}
                )

            # Query health metrics
            metrics = await self.firestore_client.query_documents(
                collection="health_data",
                filters=filters,
                limit=limit,
                offset=offset,
                order_by="created_at",
            )

            # Get total count for pagination
            total_count = await self.firestore_client.count_documents(
                collection="health_data", filters=filters
            )

            self.logger.info(
                "Retrieved user health data",
                extra={
                    "user_id": user_id,
                    "count": len(metrics),
                    "total": total_count,
                    "filters": len(filters),
                },
            )

            return {
                "metrics": metrics,
                "pagination": {
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_more": offset + len(metrics) < total_count,
                },
                "filters": {
                    "metric_type": metric_type,
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                },
            }

        except FirestoreError as e:
            self.logger.exception("Firestore error retrieving health data: %s", e)
            raise HealthDataServiceError(f"Failed to retrieve health data: {e!s}")

    async def delete_health_data(
        self, user_id: str, processing_id: str | None = None
    ) -> dict[str, Any]:
        """Delete user's health data with audit trail.

        Args:
            user_id: User ID to delete data for
            processing_id: Optional specific processing job to delete

        Returns:
            Deletion summary

        Raises:
            HealthDataServiceError: If deletion operation fails
        """
        try:
            deleted_count = 0

            if processing_id:
                # Delete specific processing job and related metrics
                filters = [
                    {"field": "user_id", "op": "==", "value": user_id},
                    {"field": "processing_id", "op": "==", "value": processing_id},
                ]
                deleted_count = await self.firestore_client.delete_documents(
                    collection="health_data", filters=filters
                )

                # Delete processing record
                await self.firestore_client.delete_document(
                    collection="processing_jobs", document_id=processing_id
                )

            else:
                # Delete all user data
                filters = [{"field": "user_id", "op": "==", "value": user_id}]
                deleted_count = await self.firestore_client.delete_documents(
                    collection="health_data", filters=filters
                )

                # Delete all processing jobs
                await self.firestore_client.delete_documents(
                    collection="processing_jobs", filters=filters
                )

            # Create audit log
            audit_record = {
                "user_id": user_id,
                "action": "data_deletion",
                "processing_id": processing_id,
                "deleted_metrics": deleted_count,
                "timestamp": datetime.now(UTC).isoformat(),
                "reason": "user_request",
            }

            await self.firestore_client.create_document(
                collection="audit_logs", data=audit_record
            )

            self.logger.info(
                "User health data deleted",
                extra={
                    "user_id": user_id,
                    "processing_id": processing_id,
                    "deleted_count": deleted_count,
                },
            )

            return {
                "deleted_metrics": deleted_count,
                "processing_id": processing_id,
                "timestamp": audit_record["timestamp"],
            }

        except FirestoreError as e:
            self.logger.exception("Firestore error during data deletion: %s", e)
            raise HealthDataServiceError(f"Failed to delete health data: {e!s}")

    def _validate_metric_business_rules(self, metric: HealthMetric) -> bool:
        """Validate health metric against business rules."""
        try:
            # Check for required metric type
            if not metric.metric_type:
                return False

            # Validate biometric data ranges
            if (
                metric.metric_type.value in ["heart_rate", "blood_pressure"]
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
            if (
                metric.metric_type.value == "mood_assessment"
                and metric.mental_health_data is not None
            ):
                return True

            return False

        except Exception as e:
            logger.warning("Business rule validation failed: %s", e)
            return False

    async def _store_metrics(
        self, user_id: str, metrics: list[HealthMetric], processing_id: str
    ) -> None:
        """Store validated health metrics to Firestore.

        Args:
            user_id: User identifier
            metrics: List of validated health metrics
            processing_id: Processing job identifier
        """
        # Store metrics in batch for better performance
        for metric in metrics:
            metric_data = {
                "user_id": user_id,
                "metric_id": str(metric.metric_id),
                "metric_type": metric.metric_type.value,
                "data": metric.model_dump(),
                "processing_id": processing_id,
                "created_at": metric.created_at,
                "device_id": metric.device_id or "unknown",
            }

            # Validate metric data before storing
            if not metric.metric_type:
                logger.warning("Metric missing type, skipping: %s", metric.metric_id)
                continue

            if (
                metric.metric_type.value in {"heart_rate", "blood_pressure"}
                and not metric.biometric_data
            ):
                logger.warning("Biometric metric missing data: %s", metric.metric_id)
                continue

            try:
                # Create individual documents for each metric
                await self.firestore_client.create_document(
                    collection="health_metrics", data=metric_data
                )
            except FirestoreError as e:
                logger.exception("Error storing metric: %s", e)

    def _calculate_progress(self, processing_doc: dict[str, Any]) -> float:
        """Calculate processing progress percentage."""
        status = processing_doc.get("status")

        if status == ProcessingStatus.RECEIVED.value:
            return 10.0
        if status == ProcessingStatus.PROCESSING.value:
            return 50.0
        if status == ProcessingStatus.COMPLETED.value:
            return 100.0
        if status == ProcessingStatus.FAILED.value:
            return 0.0
        return 0.0
