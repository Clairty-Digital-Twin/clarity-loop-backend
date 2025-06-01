"""
CLARITY Digital Twin Platform - Health Data Service.

Business logic layer for processing, validating, and managing health data
with HIPAA-compliant audit trails and comprehensive error handling.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Any

from pydantic import ValidationError

from ..models.health_data import (
    HealthDataUpload,
    HealthDataResponse, 
    HealthMetric,
    ProcessingStatus,
)
from ..storage.firestore_client import (
    FirestoreClient,
    FirestoreError,
    DocumentNotFoundError,
)

# Configure logger
logger = logging.getLogger(__name__)


class HealthDataServiceError(Exception):
    """Base exception for health data service errors."""
    pass


class DataValidationError(HealthDataServiceError):
    """Raised when health data validation fails."""
    pass


class DataNotFoundError(HealthDataServiceError):
    """Raised when requested health data is not found."""
    pass


class HealthDataService:
    """
    Service layer for health data processing and management.
    
    Handles business logic for:
    - Health data validation and processing
    - Async processing job management
    - Data retrieval with filtering and pagination
    - HIPAA-compliant audit logging
    """

    def __init__(self, firestore_client: FirestoreClient):
        """
        Initialize health data service.
        
        Args:
            firestore_client: Configured Firestore client for data persistence
        """
        self.firestore_client = firestore_client
        self.logger = logging.getLogger(__name__)

    async def process_health_data(self, upload_data: HealthDataUpload) -> HealthDataResponse:
        """Process health data upload with validation and storage."""
        try:
            processing_id = str(uuid.uuid4())
            current_time = datetime.now(timezone.utc)
            
            # Validate metrics using existing Pydantic validation
            valid_metrics = []
            validation_errors = []
            
            for metric in upload_data.metrics:
                try:
                    # The metric is already validated by Pydantic, but we can add business rules
                    if self._validate_metric_business_rules(metric):
                        valid_metrics.append(metric)
                    else:
                        validation_errors.append(f"Metric {metric.metric_id} failed business validation")
                except Exception as e:
                    validation_errors.append(f"Metric {metric.metric_id}: {str(e)}")
            
            # Create processing job document
            processing_doc = {
                "processing_id": processing_id,
                "user_id": str(upload_data.user_id),
                "status": ProcessingStatus.RECEIVED.value,
                "upload_source": upload_data.upload_source,
                "client_timestamp": upload_data.client_timestamp.isoformat(),
                "server_timestamp": current_time.isoformat(),
                "total_metrics": len(upload_data.metrics),
                "accepted_metrics": len(valid_metrics),
                "rejected_metrics": len(upload_data.metrics) - len(valid_metrics),
                "validation_errors": validation_errors,
                "sync_token": upload_data.sync_token,
                "created_at": current_time.isoformat(),
                "updated_at": current_time.isoformat()
            }
            
            # Store processing job
            await self.firestore_client.create_document(
                collection="processing_jobs",
                document_id=processing_id,
                data=processing_doc
            )
            
            # Store valid metrics
            if valid_metrics:
                await self._store_metrics(upload_data.user_id, valid_metrics, processing_id)
            
            # Update processing status
            await self.firestore_client.update_document(
                collection="processing_jobs",
                document_id=processing_id,
                data={
                    "status": ProcessingStatus.PROCESSING.value,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Log operation
            self.logger.info(f"Health data processing initiated: {processing_id}")
            
            return HealthDataResponse(
                processing_id=uuid.UUID(processing_id),
                status=ProcessingStatus.PROCESSING,
                accepted_metrics=len(valid_metrics),
                rejected_metrics=len(validation_errors),
                validation_errors=[],  # Return empty for now, could enhance later
                estimated_processing_time=len(valid_metrics) * 2,  # 2 seconds per metric
                sync_token=upload_data.sync_token,
                message="Health data processing started",
                timestamp=current_time
            )
            
        except Exception as e:
            self.logger.error(f"Unexpected error during health data processing: {e}")
            raise HealthDataServiceError(f"Health data processing failed: {str(e)}")

    async def get_processing_status(self, processing_id: str, user_id: str) -> dict[str, Any]:
        """
        Get processing status for a health data upload job.
        
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
                collection="processing_jobs",
                document_id=processing_id
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
                "completed_at": doc.get("completed_at")
            }
            
        except DocumentNotFoundError:
            raise DataNotFoundError(f"Processing job {processing_id} not found")
        except FirestoreError as e:
            self.logger.error(f"Firestore error retrieving processing status: {e}")
            raise HealthDataServiceError(f"Failed to retrieve processing status: {str(e)}")

    async def get_user_health_data(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        metric_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> dict[str, Any]:
        """
        Retrieve user's health data with filtering and pagination.
        
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
            filters = [("user_id", "==", user_id)]
            
            if metric_type:
                filters.append(("type", "==", metric_type))
            
            if start_date:
                filters.append(("timestamp", ">=", start_date.isoformat()))
                
            if end_date:
                filters.append(("timestamp", "<=", end_date.isoformat()))
            
            # Query health metrics
            metrics = await self.firestore_client.query_documents(
                collection="health_metrics",
                filters=filters,
                limit=limit,
                offset=offset,
                order_by="timestamp"
            )
            
            # Get total count for pagination
            total_count = await self.firestore_client.count_documents(
                collection="health_metrics",
                filters=filters
            )
            
            self.logger.info(
                "Retrieved user health data",
                extra={
                    "user_id": user_id,
                    "count": len(metrics),
                    "total": total_count,
                    "filters": len(filters)
                }
            )
            
            return {
                "metrics": metrics,
                "pagination": {
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_more": offset + len(metrics) < total_count
                },
                "filters": {
                    "metric_type": metric_type,
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None
                }
            }
            
        except FirestoreError as e:
            self.logger.error(f"Firestore error retrieving health data: {e}")
            raise HealthDataServiceError(f"Failed to retrieve health data: {str(e)}")

    async def delete_health_data(self, user_id: str, processing_id: Optional[str] = None) -> dict[str, Any]:
        """
        Delete user's health data with audit trail.
        
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
                    ("user_id", "==", user_id),
                    ("processing_id", "==", processing_id)
                ]
                deleted_count = await self.firestore_client.delete_documents(
                    collection="health_metrics",
                    filters=filters
                )
                
                # Delete processing record
                await self.firestore_client.delete_document(
                    collection="processing_jobs",
                    document_id=processing_id
                )
                
            else:
                # Delete all user data
                filters = [("user_id", "==", user_id)]
                deleted_count = await self.firestore_client.delete_documents(
                    collection="health_metrics",
                    filters=filters
                )
                
                # Delete all processing jobs
                await self.firestore_client.delete_documents(
                    collection="processing_jobs",
                    filters=filters
                )
            
            # Create audit log
            audit_record = {
                "user_id": user_id,
                "action": "data_deletion",
                "processing_id": processing_id,
                "deleted_metrics": deleted_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": "user_request"
            }
            
            await self.firestore_client.create_document(
                collection="audit_logs",
                data=audit_record
            )
            
            self.logger.info(
                "User health data deleted",
                extra={
                    "user_id": user_id,
                    "processing_id": processing_id,
                    "deleted_count": deleted_count
                }
            )
            
            return {
                "deleted_metrics": deleted_count,
                "processing_id": processing_id,
                "timestamp": audit_record["timestamp"]
            }
            
        except FirestoreError as e:
            self.logger.error(f"Firestore error during data deletion: {e}")
            raise HealthDataServiceError(f"Failed to delete health data: {str(e)}")

    def _validate_metric_business_rules(self, metric: HealthMetric) -> bool:
        """
        Validate individual health metric against business rules.
        
        Args:
            metric: Health metric to validate
            
        Returns:
            True if metric is valid, False otherwise
        """
        try:
            # Basic validation
            if not metric.timestamp:
                return False
                
            if metric.value is not None and metric.value < 0:
                return False
                
            # Type-specific validation
            if metric.type.value.startswith("heart_rate") and metric.value:
                return 20 <= metric.value <= 300  # Reasonable HR range
                
            if metric.type.value.startswith("blood_pressure") and metric.value:
                return 50 <= metric.value <= 250  # Reasonable BP range
                
            return True
            
        except Exception:
            return False

    async def _store_metrics(self, user_id: str, metrics: list[HealthMetric], processing_id: str) -> None:
        """
        Store validated health metrics to Firestore.
        
        Args:
            user_id: User ID
            metrics: List of validated metrics
            processing_id: Processing job ID
        """
        documents = []
        
        for metric in metrics:
            doc_data = {
                "user_id": user_id,
                "processing_id": processing_id,
                "type": metric.type.value,
                "value": metric.value,
                "unit": metric.unit,
                "timestamp": metric.timestamp.isoformat(),
                "source": metric.source,
                "metadata": metric.metadata or {},
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            documents.append(doc_data)
        
        # Batch write for performance
        await self.firestore_client.batch_create_documents(
            collection="health_metrics",
            documents=documents
        )

    def _calculate_progress(self, processing_doc: dict[str, Any]) -> float:
        """Calculate processing progress percentage."""
        status = processing_doc.get("status")
        
        if status == ProcessingStatus.PENDING.value:
            return 0.0
        elif status == ProcessingStatus.PROCESSING.value:
            return 50.0
        elif status == ProcessingStatus.COMPLETED.value:
            return 100.0
        elif status == ProcessingStatus.FAILED.value:
            return 100.0
        else:
            return 0.0
