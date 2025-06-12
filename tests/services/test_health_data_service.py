"""Comprehensive tests for Health Data Service."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
import uuid

import pytest

from clarity.models.health_data import (
    ActivityData,
    BiometricData,
    HealthDataResponse,
    HealthDataUpload,
    HealthMetric,
    HealthMetricType,
    MentalHealthIndicator,
    ProcessingStatus,
    SleepData,
)
from clarity.ports.data_ports import IHealthDataRepository
from clarity.services.health_data_service import (
    DataNotFoundError,
    HealthDataService,
    HealthDataServiceError,
    MLPredictionError,
)
from clarity.services.s3_storage_service import S3StorageService


@pytest.fixture
def mock_repository():
    """Mock health data repository."""
    mock = Mock(spec=IHealthDataRepository)
    mock.save_health_data = AsyncMock(return_value=True)
    mock.get_processing_status = AsyncMock(return_value=None)
    mock.get_user_health_data = AsyncMock(return_value={"metrics": []})
    mock.delete_health_data = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_cloud_storage():
    """Mock cloud storage service."""
    mock = Mock(spec=S3StorageService)
    mock.upload_raw_health_data = AsyncMock(return_value="s3://bucket/file.json")
    mock.upload_file = AsyncMock(return_value="s3://bucket/file.json")
    return mock


@pytest.fixture
def health_data_service(mock_repository, mock_cloud_storage):
    """Create health data service with mocked dependencies."""
    return HealthDataService(
        repository=mock_repository,
        cloud_storage=mock_cloud_storage,
    )


@pytest.fixture
def health_data_service_no_storage(mock_repository):
    """Create health data service without cloud storage."""
    return HealthDataService(
        repository=mock_repository,
        cloud_storage=None,
    )


@pytest.fixture
def valid_health_data():
    """Create valid health data upload."""
    return HealthDataUpload(
        user_id=uuid.uuid4(),
        upload_source="mobile_app",
        client_timestamp=datetime.now(UTC),
        sync_token="sync-123",
        metrics=[
            HealthMetric(
                metric_id=uuid.uuid4(),
                metric_type=HealthMetricType.HEART_RATE,
                created_at=datetime.now(UTC),
                device_id="device-123",
                biometric_data=BiometricData(
                    heart_rate=72,
                    systolic_bp=120,
                    diastolic_bp=80,
                ),
            ),
            HealthMetric(
                metric_id=uuid.uuid4(),
                metric_type=HealthMetricType.ACTIVITY_LEVEL,
                created_at=datetime.now(UTC),
                device_id="device-123",
                activity_data=ActivityData(
                    steps=5000,
                    distance=3.2,
                    calories_burned=250,
                ),
            ),
        ],
    )


class TestHealthDataServiceInit:
    """Test HealthDataService initialization."""

    def test_init_with_cloud_storage(self, mock_repository, mock_cloud_storage):
        """Test initialization with cloud storage."""
        service = HealthDataService(
            repository=mock_repository,
            cloud_storage=mock_cloud_storage,
        )

        assert service.repository == mock_repository
        assert service.cloud_storage == mock_cloud_storage
        assert service.raw_data_bucket == "clarity-healthkit-raw-data"

    def test_init_without_cloud_storage(self, mock_repository):
        """Test initialization without cloud storage."""
        service = HealthDataService(
            repository=mock_repository,
            cloud_storage=None,
        )

        assert service.repository == mock_repository
        assert service.cloud_storage is None

    @patch.dict("os.environ", {"HEALTHKIT_RAW_BUCKET": "custom-bucket"})
    def test_init_with_env_vars(self, mock_repository):
        """Test initialization with environment variables."""
        service = HealthDataService(repository=mock_repository)

        assert service.raw_data_bucket == "custom-bucket"


class TestProcessHealthData:
    """Test health data processing functionality."""

    @pytest.mark.asyncio
    async def test_process_health_data_success(
        self, health_data_service, mock_repository, valid_health_data
    ):
        """Test successful health data processing."""
        with patch("uuid.uuid4", return_value="test-process-id"):
            response = await health_data_service.process_health_data(valid_health_data)

        assert isinstance(response, HealthDataResponse)
        assert str(response.processing_id) == "test-process-id"
        assert response.status == ProcessingStatus.PROCESSING
        assert response.accepted_metrics == 2
        assert response.rejected_metrics == 0
        assert response.validation_errors == []
        assert response.sync_token == "sync-123"

        # Verify repository was called
        mock_repository.save_health_data.assert_called_once()
        call_args = mock_repository.save_health_data.call_args[1]
        assert call_args["user_id"] == str(valid_health_data.user_id)
        assert call_args["processing_id"] == "test-process-id"
        assert len(call_args["metrics"]) == 2

    @pytest.mark.asyncio
    async def test_process_health_data_validation_failure(
        self, health_data_service, valid_health_data
    ):
        """Test health data processing with validation failure."""
        # Create metric with missing required fields
        invalid_metric = HealthMetric(
            metric_id=uuid.uuid4(),
            metric_type=None,  # Invalid: missing metric type
            created_at=datetime.now(UTC),
        )
        valid_health_data.metrics.append(invalid_metric)

        with pytest.raises(HealthDataServiceError) as exc_info:
            await health_data_service.process_health_data(valid_health_data)

        assert "validation failed" in str(exc_info.value).lower()
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_process_health_data_repository_error(
        self, health_data_service, mock_repository, valid_health_data
    ):
        """Test health data processing with repository error."""
        mock_repository.save_health_data.side_effect = Exception("Database error")

        with pytest.raises(HealthDataServiceError) as exc_info:
            await health_data_service.process_health_data(valid_health_data)

        assert "processing failed" in str(exc_info.value).lower()
        assert exc_info.value.status_code == 500


class TestUploadRawData:
    """Test raw data upload functionality."""

    @pytest.mark.asyncio
    async def test_upload_raw_data_s3_service(
        self, health_data_service, mock_cloud_storage, valid_health_data
    ):
        """Test raw data upload with S3StorageService."""
        user_id = str(uuid.uuid4())
        processing_id = str(uuid.uuid4())

        result = await health_data_service._upload_raw_data_to_s3(
            user_id, processing_id, valid_health_data
        )

        assert result == "s3://bucket/file.json"
        mock_cloud_storage.upload_raw_health_data.assert_called_once_with(
            user_id=user_id,
            processing_id=processing_id,
            health_data=valid_health_data,
        )

    @pytest.mark.asyncio
    async def test_upload_raw_data_generic_storage(
        self, health_data_service, mock_cloud_storage, valid_health_data
    ):
        """Test raw data upload with generic cloud storage."""
        # Make cloud storage not an S3StorageService
        mock_cloud_storage.__class__.__name__ = "GenericStorage"
        user_id = str(uuid.uuid4())
        processing_id = str(uuid.uuid4())

        result = await health_data_service._upload_raw_data_to_s3(
            user_id, processing_id, valid_health_data
        )

        assert result == "s3://bucket/file.json"
        mock_cloud_storage.upload_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_raw_data_no_storage(
        self, health_data_service_no_storage, valid_health_data
    ):
        """Test raw data upload without cloud storage."""
        user_id = str(uuid.uuid4())
        processing_id = str(uuid.uuid4())

        result = await health_data_service_no_storage._upload_raw_data_to_s3(
            user_id, processing_id, valid_health_data
        )

        assert result == f"local://{user_id}/{processing_id}.json"

    @pytest.mark.asyncio
    async def test_upload_raw_data_error(
        self, health_data_service, mock_cloud_storage, valid_health_data
    ):
        """Test raw data upload with error."""
        mock_cloud_storage.upload_raw_health_data.side_effect = Exception("S3 error")

        with pytest.raises(HealthDataServiceError) as exc_info:
            await health_data_service._upload_raw_data_to_s3(
                "user-123", "proc-123", valid_health_data
            )

        assert "S3 upload failed" in str(exc_info.value)


class TestValidateHealthMetrics:
    """Test health metrics validation."""

    def test_validate_metrics_success(self, health_data_service, valid_health_data):
        """Test successful metrics validation."""
        errors = health_data_service._validate_health_metrics(valid_health_data.metrics)
        assert errors == []

    def test_validate_metrics_missing_type(self, health_data_service):
        """Test validation with missing metric type."""
        metric = HealthMetric(
            metric_id=uuid.uuid4(),
            metric_type=None,
            created_at=datetime.now(UTC),
        )

        errors = health_data_service._validate_health_metrics([metric])
        assert len(errors) == 1
        assert "missing required fields" in errors[0]

    def test_validate_metrics_missing_created_at(self, health_data_service):
        """Test validation with missing created_at."""
        metric = HealthMetric(
            metric_id=uuid.uuid4(),
            metric_type=HealthMetricType.HEART_RATE,
            created_at=None,
        )

        errors = health_data_service._validate_health_metrics([metric])
        assert len(errors) == 1
        assert "missing required fields" in errors[0]

    def test_validate_metrics_business_rule_failure(self, health_data_service):
        """Test validation with business rule failure."""
        # Heart rate metric without biometric data
        metric = HealthMetric(
            metric_id=uuid.uuid4(),
            metric_type=HealthMetricType.HEART_RATE,
            created_at=datetime.now(UTC),
            biometric_data=None,  # Should have biometric data
        )

        errors = health_data_service._validate_health_metrics([metric])
        assert len(errors) == 1
        assert "failed business validation" in errors[0]

    def test_validate_metrics_exception(self, health_data_service):
        """Test validation with exception during processing."""
        metric = Mock()
        metric.metric_id = "test-id"
        metric.metric_type = Mock(side_effect=ValueError("Invalid metric"))

        errors = health_data_service._validate_health_metrics([metric])
        assert len(errors) == 1
        assert "Invalid metric" in errors[0]


class TestGetProcessingStatus:
    """Test processing status retrieval."""

    @pytest.mark.asyncio
    async def test_get_processing_status_success(
        self, health_data_service, mock_repository
    ):
        """Test successful status retrieval."""
        status_data = {
            "processing_id": "test-123",
            "status": "completed",
            "metrics_processed": 10,
        }
        mock_repository.get_processing_status.return_value = status_data

        result = await health_data_service.get_processing_status(
            "test-123", "user-123"
        )

        assert result == status_data
        mock_repository.get_processing_status.assert_called_once_with(
            processing_id="test-123",
            user_id="user-123",
        )

    @pytest.mark.asyncio
    async def test_get_processing_status_not_found(
        self, health_data_service, mock_repository
    ):
        """Test status retrieval when not found."""
        mock_repository.get_processing_status.return_value = None

        with pytest.raises(DataNotFoundError) as exc_info:
            await health_data_service.get_processing_status("missing-123", "user-123")

        assert "missing-123" in str(exc_info.value)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_processing_status_repository_error(
        self, health_data_service, mock_repository
    ):
        """Test status retrieval with repository error."""
        mock_repository.get_processing_status.side_effect = Exception("DB error")

        with pytest.raises(HealthDataServiceError) as exc_info:
            await health_data_service.get_processing_status("test-123", "user-123")

        assert "Failed to get processing status" in str(exc_info.value)


class TestGetUserHealthData:
    """Test user health data retrieval."""

    @pytest.mark.asyncio
    async def test_get_user_health_data_success(
        self, health_data_service, mock_repository
    ):
        """Test successful health data retrieval."""
        health_data = {
            "metrics": [
                {"id": "1", "type": "heart_rate", "value": 72},
                {"id": "2", "type": "steps", "value": 5000},
            ],
            "total": 2,
        }
        mock_repository.get_user_health_data.return_value = health_data

        result = await health_data_service.get_user_health_data(
            user_id="user-123",
            limit=10,
            offset=0,
            metric_type="heart_rate",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 31, tzinfo=UTC),
        )

        assert result == health_data
        assert len(result["metrics"]) == 2

        mock_repository.get_user_health_data.assert_called_once_with(
            user_id="user-123",
            limit=10,
            offset=0,
            metric_type="heart_rate",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 31, tzinfo=UTC),
        )

    @pytest.mark.asyncio
    async def test_get_user_health_data_error(
        self, health_data_service, mock_repository
    ):
        """Test health data retrieval with error."""
        mock_repository.get_user_health_data.side_effect = Exception("DB error")

        with pytest.raises(HealthDataServiceError) as exc_info:
            await health_data_service.get_user_health_data("user-123")

        assert "Failed to retrieve health data" in str(exc_info.value)


class TestDeleteHealthData:
    """Test health data deletion."""

    @pytest.mark.asyncio
    async def test_delete_health_data_success(
        self, health_data_service, mock_repository
    ):
        """Test successful health data deletion."""
        result = await health_data_service.delete_health_data(
            user_id="user-123",
            processing_id="proc-123",
        )

        assert result is True
        mock_repository.delete_health_data.assert_called_once_with(
            user_id="user-123",
            processing_id="proc-123",
        )

    @pytest.mark.asyncio
    async def test_delete_health_data_all_user_data(
        self, health_data_service, mock_repository
    ):
        """Test deletion of all user data."""
        result = await health_data_service.delete_health_data(
            user_id="user-123",
            processing_id=None,
        )

        assert result is True
        mock_repository.delete_health_data.assert_called_once_with(
            user_id="user-123",
            processing_id=None,
        )

    @pytest.mark.asyncio
    async def test_delete_health_data_error(
        self, health_data_service, mock_repository
    ):
        """Test health data deletion with error."""
        mock_repository.delete_health_data.side_effect = Exception("DB error")

        with pytest.raises(HealthDataServiceError) as exc_info:
            await health_data_service.delete_health_data("user-123")

        assert "Failed to delete health data" in str(exc_info.value)


class TestValidateMetricBusinessRules:
    """Test metric business rules validation."""

    def test_validate_business_rules_heart_rate_valid(self, health_data_service):
        """Test valid heart rate metric."""
        metric = HealthMetric(
            metric_id=uuid.uuid4(),
            metric_type=HealthMetricType.HEART_RATE,
            created_at=datetime.now(UTC),
            biometric_data=BiometricData(heart_rate=72),
        )

        assert health_data_service._validate_metric_business_rules(metric) is True

    def test_validate_business_rules_heart_rate_no_data(self, health_data_service):
        """Test heart rate metric without biometric data."""
        metric = HealthMetric(
            metric_id=uuid.uuid4(),
            metric_type=HealthMetricType.HEART_RATE,
            created_at=datetime.now(UTC),
            biometric_data=None,
        )

        assert health_data_service._validate_metric_business_rules(metric) is False

    def test_validate_business_rules_sleep_valid(self, health_data_service):
        """Test valid sleep metric."""
        metric = HealthMetric(
            metric_id=uuid.uuid4(),
            metric_type=HealthMetricType.SLEEP_ANALYSIS,
            created_at=datetime.now(UTC),
            sleep_data=SleepData(
                sleep_start=datetime.now(UTC),
                sleep_end=datetime.now(UTC),
                sleep_stages=[],
            ),
        )

        assert health_data_service._validate_metric_business_rules(metric) is True

    def test_validate_business_rules_activity_valid(self, health_data_service):
        """Test valid activity metric."""
        metric = HealthMetric(
            metric_id=uuid.uuid4(),
            metric_type=HealthMetricType.ACTIVITY_LEVEL,
            created_at=datetime.now(UTC),
            activity_data=ActivityData(steps=5000),
        )

        assert health_data_service._validate_metric_business_rules(metric) is True

    def test_validate_business_rules_mood_valid(self, health_data_service):
        """Test valid mood assessment metric."""
        metric = HealthMetric(
            metric_id=uuid.uuid4(),
            metric_type=HealthMetricType.MOOD_ASSESSMENT,
            created_at=datetime.now(UTC),
            mental_health_data=MentalHealthIndicator(
                mood_score=7,
                stress_level=3,
            ),
        )

        assert health_data_service._validate_metric_business_rules(metric) is True

    def test_validate_business_rules_no_metric_type(self, health_data_service):
        """Test metric without type."""
        metric = HealthMetric(
            metric_id=uuid.uuid4(),
            metric_type=None,
            created_at=datetime.now(UTC),
        )

        assert health_data_service._validate_metric_business_rules(metric) is False

    def test_validate_business_rules_exception(self, health_data_service):
        """Test validation with exception."""
        metric = Mock()
        metric.metric_type = Mock(side_effect=AttributeError("Invalid"))

        assert health_data_service._validate_metric_business_rules(metric) is False


class TestExceptionClasses:
    """Test custom exception classes."""

    def test_health_data_service_error(self):
        """Test HealthDataServiceError."""
        error = HealthDataServiceError("Test error", status_code=400)
        assert str(error) == "Test error"
        assert error.status_code == 400

    def test_health_data_service_error_default_status(self):
        """Test HealthDataServiceError with default status."""
        error = HealthDataServiceError("Test error")
        assert error.status_code == 500

    def test_data_not_found_error(self):
        """Test DataNotFoundError."""
        error = DataNotFoundError("Data not found")
        assert str(error) == "Data not found"
        assert error.status_code == 404
        assert isinstance(error, HealthDataServiceError)

    def test_ml_prediction_error(self):
        """Test MLPredictionError."""
        error = MLPredictionError("Prediction failed", model_name="test_model")
        assert "ML Prediction Error in test_model" in str(error)
        assert error.status_code == 503
        assert error.model_name == "test_model"

    def test_ml_prediction_error_no_model(self):
        """Test MLPredictionError without model name."""
        error = MLPredictionError("Prediction failed")
        assert str(error) == "ML Prediction Error: Prediction failed"
        assert error.model_name is None