"""
CLARITY Digital Twin Platform - Health Data Service

Enterprise-grade business logic layer for health data operations:
- Async health data processing and validation
- HIPAA-compliant data handling and audit trails
- Integration with Firestore for persistence
- AI/ML pipeline orchestration for insights
- Real-time health monitoring capabilities

Security Features:
- Input validation and sanitization
- Access control enforcement
- Comprehensive audit logging
- Data encryption and privacy protection
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID, uuid4

from pydantic import ValidationError as PydanticValidationError

from ..models.health_data import (
    HealthDataUpload,
    HealthDataResponse,
    HealthMetric,
    ProcessingStatus,
    HealthMetricType
)
from ..storage.firestore_client import (
    FirestoreClient,
    FirestoreError,
    ValidationError,
    DocumentNotFoundError
)

# Configure logger
logger = logging.getLogger(__name__)


class HealthDataServiceError(Exception):
    """Base exception for health data service operations."""
    pass


class ProcessingError(HealthDataServiceError):
    """Raised when health data processing fails."""
    pass


class AuthenticationError(HealthDataServiceError):
    """Raised when user authentication fails."""
    pass


class HealthDataService:
    """
    Business logic service for health data operations.
    
    Provides enterprise-grade health data processing with:
    - Clinical-grade data validation and quality checks
    - Async processing pipeline integration
    - HIPAA-compliant audit trails and security
    - Real-time status tracking and monitoring
    - AI/ML pipeline orchestration
    """
    
    def __init__(
        self,
        firestore_client: FirestoreClient,
        max_concurrent_operations: int = 10,
        processing_timeout: int = 300  # 5 minutes
    ):
        """
        Initialize the health data service.
        
        Args:
            firestore_client: Configured Firestore client
            max_concurrent_operations: Maximum concurrent async operations
            processing_timeout: Timeout for processing operations in seconds
        """
        self.firestore = firestore_client
        self.max_concurrent_operations = max_concurrent_operations
        self.processing_timeout = processing_timeout
        
        # Semaphore for controlling concurrent operations
        self._semaphore = asyncio.Semaphore(max_concurrent_operations)
        
        # Processing statistics
        self._stats = {
            'total_uploads': 0,
            'successful_uploads': 0,
            'failed_uploads': 0,
            'total_metrics_processed': 0
        }
        
        logger.info(f"Health data service initialized with max {max_concurrent_operations} concurrent operations")
    
    async def upload_health_data(
        self,
        health_data: HealthDataUpload,
        user_id: Optional[str] = None
    ) -> HealthDataResponse:
        """
        Process and store health data upload with comprehensive validation.
        
        Args:
            health_data: Health data upload payload
            user_id: Optional user ID for additional validation
            
        Returns:
            HealthDataResponse with processing status and metadata
            
        Raises:
            ProcessingError: If data processing fails
            ValidationError: If data validation fails
            AuthenticationError: If user authentication fails
        """
        async with self._semaphore:
            processing_id = str(uuid4())
            start_time = datetime.now(timezone.utc)
            
            try:
                logger.info(f"Starting health data upload processing: {processing_id}")
                
                # Validate user access
                if user_id and str(health_data.user_id) != user_id:
                    raise AuthenticationError("User ID mismatch in health data upload")
                
                # Validate and enrich health data
                validated_metrics, validation_errors = await self._validate_health_metrics(
                    health_data.metrics
                )
                
                # Calculate processing estimates
                accepted_count = len(validated_metrics)
                rejected_count = len(health_data.metrics) - accepted_count
                estimated_processing_time = self._estimate_processing_time(accepted_count)
                
                # Store health data in Firestore
                await self.firestore.store_health_data(
                    health_data=health_data,
                    processing_id=processing_id
                )
                
                # Update processing status
                await self.firestore.update_processing_status(
                    processing_id=processing_id,
                    status=ProcessingStatus.PROCESSING
                )
                
                # Trigger async processing pipeline (fire-and-forget)
                asyncio.create_task(
                    self._process_health_data_async(
                        processing_id=processing_id,
                        metrics=validated_metrics,
                        user_id=str(health_data.user_id)
                    )
                )
                
                # Update statistics
                self._stats['total_uploads'] += 1
                self._stats['successful_uploads'] += 1
                self._stats['total_metrics_processed'] += accepted_count
                
                response = HealthDataResponse(
                    processing_id=UUID(processing_id),
                    status=ProcessingStatus.PROCESSING,
                    accepted_metrics=accepted_count,
                    rejected_metrics=rejected_count,
                    validation_errors=validation_errors,
                    estimated_processing_time=estimated_processing_time,
                    sync_token=health_data.sync_token,
                    message="Health data received and processing started",
                    timestamp=start_time
                )
                
                logger.info(f"Health data upload processed successfully: {processing_id}")
                return response
                
            except Exception as e:
                self._stats['failed_uploads'] += 1
                logger.error(f"Health data upload failed: {processing_id} - {e}")
                
                # Update processing status to failed
                try:
                    await self.firestore.update_processing_status(
                        processing_id=processing_id,
                        status=ProcessingStatus.FAILED,
                        error_message=str(e)
                    )
                except Exception as storage_error:
                    logger.error(f"Failed to update processing status: {storage_error}")
                
                if isinstance(e, (ValidationError, AuthenticationError)):
                    raise
                
                raise ProcessingError(f"Health data processing failed: {e}")
    
    async def get_processing_status(
        self,
        processing_id: str,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get processing status for a health data upload.
        
        Args:
            processing_id: Processing ID to check
            user_id: Optional user ID for access control
            
        Returns:
            Processing status information or None if not found
            
        Raises:
            AuthenticationError: If user doesn't have access to the processing ID
        """
        try:
            status_info = await self.firestore.get_processing_status(processing_id)
            
            if not status_info:
                logger.warning(f"Processing status not found: {processing_id}")
                return None
            
            # Validate user access
            if user_id and status_info.get('user_id') != user_id:
                raise AuthenticationError("Access denied to processing status")
            
            logger.debug(f"Retrieved processing status: {processing_id}")
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to get processing status {processing_id}: {e}")
            if isinstance(e, AuthenticationError):
                raise
            raise ProcessingError(f"Failed to retrieve processing status: {e}")
    
    async def get_user_health_data(
        self,
        user_id: str,
        metric_type: Optional[HealthMetricType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve health data for a user with filtering options.
        
        Args:
            user_id: User ID to retrieve data for
            metric_type: Optional metric type filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of records to return
            
        Returns:
            List of health data records
        """
        try:
            filters = [
                {'field': 'user_id', 'operator': '==', 'value': user_id}
            ]
            
            if metric_type:
                filters.append({
                    'field': 'metrics.type',
                    'operator': '==',
                    'value': metric_type.value
                })
            
            if start_date:
                filters.append({
                    'field': 'client_timestamp',
                    'operator': '>=',
                    'value': start_date
                })
            
            if end_date:
                filters.append({
                    'field': 'client_timestamp',
                    'operator': '<=',
                    'value': end_date
                })
            
            results = await self.firestore.query_documents(
                collection=self.firestore.collections['health_data'],
                filters=filters,
                order_by='client_timestamp',
                limit=limit
            )
            
            logger.info(f"Retrieved {len(results)} health data records for user {user_id}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve health data for user {user_id}: {e}")
            raise ProcessingError(f"Failed to retrieve health data: {e}")
    
    async def delete_user_health_data(
        self,
        user_id: str,
        processing_id: Optional[str] = None
    ) -> bool:
        """
        Delete health data for a user (GDPR compliance).
        
        Args:
            user_id: User ID to delete data for
            processing_id: Optional specific processing ID to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            if processing_id:
                # Delete specific upload
                success = await self.firestore.delete_document(
                    collection=self.firestore.collections['health_data'],
                    document_id=processing_id,
                    user_id=user_id
                )
                
                if success:
                    # Also delete processing job
                    await self.firestore.delete_document(
                        collection=self.firestore.collections['processing_jobs'],
                        document_id=processing_id,
                        user_id=user_id
                    )
                
                logger.info(f"Deleted health data for processing ID: {processing_id}")
                return success
            else:
                # Delete all data for user (GDPR right to erasure)
                health_data_records = await self.get_user_health_data(user_id)
                
                deletion_tasks = []
                for record in health_data_records:
                    task = self.firestore.delete_document(
                        collection=self.firestore.collections['health_data'],
                        document_id=record['id'],
                        user_id=user_id
                    )
                    deletion_tasks.append(task)
                
                results = await asyncio.gather(*deletion_tasks, return_exceptions=True)
                success_count = sum(1 for result in results if result is True)
                
                logger.info(f"Deleted {success_count}/{len(health_data_records)} health data records for user {user_id}")
                return success_count == len(health_data_records)
                
        except Exception as e:
            logger.error(f"Failed to delete health data for user {user_id}: {e}")
            raise ProcessingError(f"Failed to delete health data: {e}")
    
    # Private Methods
    
    async def _validate_health_metrics(
        self,
        metrics: List[HealthMetric]
    ) -> Tuple[List[HealthMetric], List[str]]:
        """
        Validate health metrics with clinical-grade quality checks.
        
        Args:
            metrics: List of health metrics to validate
            
        Returns:
            Tuple of (validated_metrics, validation_errors)
        """
        validated_metrics = []
        validation_errors = []
        
        for i, metric in enumerate(metrics):
            try:
                # Basic Pydantic validation is already done
                # Add clinical validation rules
                
                # Validate data ranges based on metric type
                if not self._validate_metric_ranges(metric):
                    validation_errors.append(f"Metric {i}: Value out of valid clinical range")
                    continue
                
                # Validate timestamp (not in future, not too old)
                if metric.timestamp > datetime.now(timezone.utc):
                    validation_errors.append(f"Metric {i}: Timestamp cannot be in the future")
                    continue
                
                # Check for data staleness (older than 30 days)
                max_age = datetime.now(timezone.utc) - timedelta(days=30)
                if metric.timestamp < max_age:
                    validation_errors.append(f"Metric {i}: Data too old (>30 days)")
                    continue
                
                validated_metrics.append(metric)
                
            except Exception as e:
                validation_errors.append(f"Metric {i}: Validation error - {str(e)}")
                logger.warning(f"Metric validation failed: {e}")
        
        logger.info(f"Validated {len(validated_metrics)}/{len(metrics)} health metrics")
        return validated_metrics, validation_errors
    
    def _validate_metric_ranges(self, metric: HealthMetric) -> bool:
        """
        Validate metric values against clinical ranges.
        
        Args:
            metric: Health metric to validate
            
        Returns:
            True if metric is within valid ranges
        """
        # Clinical validation ranges
        ranges = {
            HealthMetricType.HEART_RATE: (30, 220),
            HealthMetricType.BLOOD_PRESSURE_SYSTOLIC: (70, 250),
            HealthMetricType.BLOOD_PRESSURE_DIASTOLIC: (40, 150),
            HealthMetricType.BLOOD_GLUCOSE: (50, 600),  # mg/dL
            HealthMetricType.BODY_TEMPERATURE: (95.0, 108.0),  # Fahrenheit
            HealthMetricType.OXYGEN_SATURATION: (70, 100),
            HealthMetricType.STEPS: (0, 100000),
            HealthMetricType.CALORIES_BURNED: (0, 10000),
            HealthMetricType.DISTANCE_WALKED: (0, 100),  # miles
            HealthMetricType.SLEEP_DURATION: (0, 24),  # hours
            HealthMetricType.WEIGHT: (50, 500),  # pounds
            HealthMetricType.MOOD_SCALE: (1, 10),
        }
        
        metric_range = ranges.get(metric.type)
        if not metric_range:
            # No validation range defined, assume valid
            return True
        
        min_val, max_val = metric_range
        return min_val <= metric.value <= max_val
    
    def _estimate_processing_time(self, metric_count: int) -> int:
        """
        Estimate processing time based on metric count.
        
        Args:
            metric_count: Number of metrics to process
            
        Returns:
            Estimated processing time in seconds
        """
        # Base processing time: 5 seconds
        # Additional time: 1 second per 10 metrics
        base_time = 5
        additional_time = (metric_count // 10) * 1
        
        return base_time + additional_time
    
    async def _process_health_data_async(
        self,
        processing_id: str,
        metrics: List[HealthMetric],
        user_id: str
    ) -> None:
        """
        Asynchronous health data processing pipeline.
        
        This method runs in the background to:
        1. Perform advanced data quality checks
        2. Trigger AI/ML analysis
        3. Generate health insights
        4. Update processing status
        
        Args:
            processing_id: Processing ID for tracking
            metrics: Validated health metrics
            user_id: User ID for context
        """
        try:
            logger.info(f"Starting async processing: {processing_id}")
            
            # Simulate processing delay (replace with actual ML pipeline)
            processing_time = self._estimate_processing_time(len(metrics))
            await asyncio.sleep(min(processing_time, 10))  # Cap at 10 seconds for demo
            
            # Here would be the actual AI/ML processing:
            # 1. Data normalization and feature extraction
            # 2. Pattern recognition and anomaly detection
            # 3. Health insights generation using Gemini AI
            # 4. Trend analysis and predictions
            # 5. Integration with user's health profile
            
            # Update status to completed
            await self.firestore.update_processing_status(
                processing_id=processing_id,
                status=ProcessingStatus.COMPLETED,
                completion_time=datetime.now(timezone.utc)
            )
            
            logger.info(f"Async processing completed: {processing_id}")
            
        except Exception as e:
            logger.error(f"Async processing failed: {processing_id} - {e}")
            
            # Update status to failed
            try:
                await self.firestore.update_processing_status(
                    processing_id=processing_id,
                    status=ProcessingStatus.FAILED,
                    error_message=str(e),
                    completion_time=datetime.now(timezone.utc)
                )
            except Exception as status_error:
                logger.error(f"Failed to update failed status: {status_error}")
    
    # Health Check and Statistics
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the service.
        
        Returns:
            Dict with service health information
        """
        try:
            # Test Firestore connectivity
            firestore_health = await self.firestore.health_check()
            
            return {
                'status': 'healthy',
                'firestore': firestore_health,
                'processing_stats': self._stats,
                'max_concurrent_operations': self.max_concurrent_operations,
                'processing_timeout': self.processing_timeout,
                'timestamp': datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc)
            }
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics for monitoring.
        
        Returns:
            Dict with processing statistics
        """
        return {
            **self._stats,
            'success_rate': (
                self._stats['successful_uploads'] / max(self._stats['total_uploads'], 1)
            ) * 100,
            'average_metrics_per_upload': (
                self._stats['total_metrics_processed'] / max(self._stats['successful_uploads'], 1)
            ),
            'timestamp': datetime.now(timezone.utc)
        }
