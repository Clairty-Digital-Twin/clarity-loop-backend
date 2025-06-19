"""Comprehensive tests for Health Data Service."""

from __future__ import annotations

from datetime import UTC, datetime
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
    ProcessingStatus,
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
def mock_repository() -> Mock:
    """Mock health data repository."""
    mock = Mock(spec=IHealthDataRepository)
    mock.save_health_data = AsyncMock(return_value=True)
    mock.get_processing_status = AsyncMock(return_value=None)
    mock.get_user_health_data = AsyncMock(return_value={"metrics": []})
    mock.delete_health_data = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_cloud_storage() -> Mock:
    """Mock cloud storage service."""
    mock = Mock(spec=S3StorageService)
    mock.upload_raw_health_data = AsyncMock(return_value="s3://bucket/file.json")
    mock.upload_file = AsyncMock(return_value="s3://bucket/file.json")
    return mock


@pytest.fixture
def health_data_service(
    mock_repository: Mock, mock_cloud_storage: Mock
) -> HealthDataService:
    """Create health data service with mocked dependencies."""
    return HealthDataService(
        repository=mock_repository,
        cloud_storage=mock_cloud_storage,
    )


@pytest.fixture
def health_data_service_no_storage(mock_repository: Mock) -> HealthDataService:
    """Create health data service without cloud storage."""
    with patch("clarity.services.health_data_service._HAS_S3", False):  # noqa: FBT003
        return HealthDataService(
            repository=mock_repository,
            cloud_storage=None,
        )


@pytest.fixture
def valid_health_data() -> HealthDataUpload:
    """Create valid health data upload."""
    return HealthDataUpload(
        user_id=uuid.uuid4(),
        upload_source="mobile_app",
        client_timestamp=datetime.now(UTC),
        sync_token="sync-123",  # noqa: S106 - Test sync token value
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

    def test_init_with_cloud_storage(
        self, mock_repository: Mock, mock_cloud_storage: Mock
    ) -> None:
        """Test initialization with cloud storage."""
        service = HealthDataService(
            repository=mock_repository,
            cloud_storage=mock_cloud_storage,
        )

        assert service.repository == mock_repository
        assert service.cloud_storage == mock_cloud_storage
        assert service.raw_data_bucket == "clarity-healthkit-raw-data"

    @patch("clarity.services.health_data_service._HAS_S3", False)  # noqa: FBT003
    def test_init_without_cloud_storage(self, mock_repository: Mock) -> None:
        """Test initialization without cloud storage."""
        service = HealthDataService(
            repository=mock_repository,
            cloud_storage=None,
        )

        assert service.repository == mock_repository
        assert service.cloud_storage is None

    @patch.dict("os.environ", {"HEALTHKIT_RAW_BUCKET": "custom-bucket"})
    def test_init_with_env_vars(self, mock_repository: Mock) -> None:
        """Test initialization with environment variables."""
        service = HealthDataService(repository=mock_repository)

        assert service.raw_data_bucket == "custom-bucket"


class TestProcessHealthData:
    """Test health data processing functionality."""

    @pytest.mark.asyncio
    async def test_process_health_data_success(
        self,
        health_data_service: HealthDataService,
        mock_repository: Mock,
        valid_health_data: HealthDataUpload,
    ) -> None:
        """Test successful health data processing."""
        test_uuid = uuid.uuid4()
        with patch("uuid.uuid4", return_value=test_uuid):
            response = await health_data_service.process_health_data(valid_health_data)

        assert isinstance(response, HealthDataResponse)
        assert response.processing_id == test_uuid
        assert response.status == ProcessingStatus.PROCESSING
        assert response.accepted_metrics == 2
        assert response.rejected_metrics == 0
        assert response.validation_errors == []
        assert response.sync_token == "sync-123"  # noqa: S105 - Test sync token value

        # Verify repository was called
        mock_repository.save_health_data.assert_called_once()
        call_args = mock_repository.save_health_data.call_args[1]
        assert call_args["user_id"] == str(valid_health_data.user_id)
        assert call_args["processing_id"] == str(test_uuid)
        assert len(call_args["metrics"]) == 2

    @pytest.mark.asyncio
    async def test_process_health_data_validation_failure(
        self,
        health_data_service: HealthDataService,
        valid_health_data: HealthDataUpload,
    ) -> None:
        """Test health data processing with validation failure."""
        # Create an invalid health data by removing metric_type from existing metrics
        valid_health_data.metrics[0].metric_type = None

        with pytest.raises(HealthDataServiceError) as exc_info:
            await health_data_service.process_health_data(valid_health_data)

        assert "validation failed" in str(exc_info.value).lower()
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_process_health_data_repository_error(
        self,
        health_data_service: HealthDataService,
        mock_repository: Mock,
        valid_health_data: HealthDataUpload,
    ) -> None:
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
        self,
        health_data_service: HealthDataService,
        mock_cloud_storage: Mock,
        valid_health_data: HealthDataUpload,
    ) -> None:
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
        self, mock_repository: Mock, valid_health_data: HealthDataUpload
    ) -> None:
        """Test raw data upload with generic cloud storage."""
        # Create a mock that's not an S3StorageService
        mock_generic_storage = Mock()
        mock_generic_storage.upload_file = AsyncMock(
            return_value="s3://bucket/file.json"
        )

        # We need to ensure isinstance check fails
        with patch(
            "clarity.services.health_data_service.isinstance",
            side_effect=lambda _obj, _cls: False,
        ):
            service = HealthDataService(
                repository=mock_repository,
                cloud_storage=mock_generic_storage,
            )

            user_id = str(uuid.uuid4())
            processing_id = str(uuid.uuid4())

            result = await service._upload_raw_data_to_s3(
                user_id, processing_id, valid_health_data
            )

            assert result == "s3://bucket/file.json"
            mock_generic_storage.upload_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_raw_data_no_storage(
        self,
        health_data_service_no_storage: HealthDataService,
        valid_health_data: HealthDataUpload,
    ) -> None:
        """Test raw data upload without cloud storage."""
        user_id = str(uuid.uuid4())
        processing_id = str(uuid.uuid4())

        result = await health_data_service_no_storage._upload_raw_data_to_s3(
            user_id, processing_id, valid_health_data
        )

        assert result == f"local://{user_id}/{processing_id}.json"

    @pytest.mark.asyncio
    async def test_upload_raw_data_error(
        self,
        health_data_service: HealthDataService,
        mock_cloud_storage: Mock,
        valid_health_data: HealthDataUpload,
    ) -> None:
        """Test raw data upload with error."""
        mock_cloud_storage.upload_raw_health_data.side_effect = Exception("S3 error")

        with pytest.raises(HealthDataServiceError) as exc_info:
            await health_data_service._upload_raw_data_to_s3(
                "user-123", "proc-123", valid_health_data
            )

        assert "S3 upload failed" in str(exc_info.value)


class TestValidateHealthMetrics:
    """Test health metrics validation."""

    def test_validate_metrics_success(
        self,
        health_data_service: HealthDataService,
        valid_health_data: HealthDataUpload,
    ) -> None:
        """Test successful metrics validation."""
        errors = health_data_service._validate_health_metrics(valid_health_data.metrics)
        assert errors == []

    def test_validate_metrics_missing_type(
        self, health_data_service: HealthDataService
    ) -> None:
        """Test validation with missing metric type."""
        # Create a mock metric with None type to test validation
        metric = Mock()
        metric.metric_id = uuid.uuid4()
        metric.metric_type = None
        metric.created_at = datetime.now(UTC)

        errors = health_data_service._validate_health_metrics([metric])
        assert len(errors) == 1
        assert "missing required fields" in errors[0]

    def test_validate_metrics_missing_created_at(
        self, health_data_service: HealthDataService
    ) -> None:
        """Test validation with missing created_at."""
        # Create a mock metric with None created_at to test validation
        metric = Mock()
        metric.metric_id = uuid.uuid4()
        metric.metric_type = HealthMetricType.HEART_RATE
        metric.created_at = None

        errors = health_data_service._validate_health_metrics([metric])
        assert len(errors) == 1
        assert "missing required fields" in errors[0]

    def test_validate_metrics_business_rule_failure(
        self, health_data_service: HealthDataService
    ) -> None:
        """Test validation with business rule failure."""
        # Create a mock heart rate metric without biometric data
        metric = Mock()
        metric.metric_id = uuid.uuid4()
        metric.metric_type = Mock()
        metric.metric_type.value = "heart_rate"
        metric.created_at = datetime.now(UTC)
        metric.biometric_data = None  # Should have biometric data

        errors = health_data_service._validate_health_metrics([metric])
        assert len(errors) == 1
        assert "failed business validation" in errors[0]

    def test_validate_metrics_exception(
        self, health_data_service: HealthDataService
    ) -> None:
        """Test validation with exception during processing."""
        metric = Mock()
        metric.metric_id = "test-id"
        metric.metric_type = None
        metric.created_at = Mock(side_effect=AttributeError("Invalid metric"))

        errors = health_data_service._validate_health_metrics([metric])
        assert len(errors) == 1
        assert "missing required fields" in errors[0]


class TestGetProcessingStatus:
    """Test processing status retrieval."""

    @pytest.mark.asyncio
    async def test_get_processing_status_success(
        self, health_data_service: HealthDataService, mock_repository: Mock
    ) -> None:
        """Test successful status retrieval."""
        status_data = {
            "processing_id": "test-123",
            "status": "completed",
            "metrics_processed": 10,
        }
        mock_repository.get_processing_status.return_value = status_data

        result = await health_data_service.get_processing_status("test-123", "user-123")

        assert result == status_data
        mock_repository.get_processing_status.assert_called_once_with(
            processing_id="test-123",
            user_id="user-123",
        )

    @pytest.mark.asyncio
    async def test_get_processing_status_not_found(
        self, health_data_service: HealthDataService, mock_repository: Mock
    ) -> None:
        """Test status retrieval when not found."""
        mock_repository.get_processing_status.return_value = None

        with pytest.raises(DataNotFoundError) as exc_info:
            await health_data_service.get_processing_status("missing-123", "user-123")

        assert "missing-123" in str(exc_info.value)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_processing_status_repository_error(
        self, health_data_service: HealthDataService, mock_repository: Mock
    ) -> None:
        """Test status retrieval with repository error."""
        mock_repository.get_processing_status.side_effect = Exception("DB error")

        with pytest.raises(HealthDataServiceError) as exc_info:
            await health_data_service.get_processing_status("test-123", "user-123")

        assert "Failed to get processing status" in str(exc_info.value)


class TestGetUserHealthData:
    """Test user health data retrieval."""

    @pytest.mark.asyncio
    async def test_get_user_health_data_success(
        self, health_data_service: HealthDataService, mock_repository: Mock
    ) -> None:
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
        self, health_data_service: HealthDataService, mock_repository: Mock
    ) -> None:
        """Test health data retrieval with error."""
        mock_repository.get_user_health_data.side_effect = Exception("DB error")

        with pytest.raises(HealthDataServiceError) as exc_info:
            await health_data_service.get_user_health_data("user-123")

        assert "Failed to retrieve health data" in str(exc_info.value)


class TestDeleteHealthData:
    """Test health data deletion."""

    @pytest.mark.asyncio
    async def test_delete_health_data_success(
        self, health_data_service: HealthDataService, mock_repository: Mock
    ) -> None:
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
        self, health_data_service: HealthDataService, mock_repository: Mock
    ) -> None:
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
        self, health_data_service: HealthDataService, mock_repository: Mock
    ) -> None:
        """Test health data deletion with error."""
        mock_repository.delete_health_data.side_effect = Exception("DB error")

        with pytest.raises(HealthDataServiceError) as exc_info:
            await health_data_service.delete_health_data("user-123")

        assert "Failed to delete health data" in str(exc_info.value)


class TestValidateMetricBusinessRules:
    """Test metric business rules validation."""

    def test_validate_business_rules_heart_rate_valid(
        self, health_data_service: HealthDataService
    ) -> None:
        """Test valid heart rate metric."""
        metric = HealthMetric(
            metric_id=uuid.uuid4(),
            metric_type=HealthMetricType.HEART_RATE,
            created_at=datetime.now(UTC),
            biometric_data=BiometricData(heart_rate=72),
        )

        assert health_data_service._validate_metric_business_rules(metric) is True

    def test_validate_business_rules_heart_rate_no_data(
        self, health_data_service: HealthDataService
    ) -> None:
        """Test heart rate metric without biometric data."""
        # Create a mock metric since Pydantic enforces validation
        metric = Mock()
        metric.metric_type = Mock()
        metric.metric_type.value = "heart_rate"
        metric.biometric_data = None

        assert health_data_service._validate_metric_business_rules(metric) is False

    def test_validate_business_rules_sleep_valid(
        self, health_data_service: HealthDataService
    ) -> None:
        """Test valid sleep metric."""
        # Create a mock sleep metric
        metric = Mock()
        metric.metric_type = Mock()
        metric.metric_type.value = "sleep_analysis"
        metric.sleep_data = Mock()  # Non-None sleep data

        assert health_data_service._validate_metric_business_rules(metric) is True

    def test_validate_business_rules_activity_valid(
        self, health_data_service: HealthDataService
    ) -> None:
        """Test valid activity metric."""
        metric = HealthMetric(
            metric_id=uuid.uuid4(),
            metric_type=HealthMetricType.ACTIVITY_LEVEL,
            created_at=datetime.now(UTC),
            activity_data=ActivityData(steps=5000),
        )

        assert health_data_service._validate_metric_business_rules(metric) is True

    def test_validate_business_rules_mood_valid(
        self, health_data_service: HealthDataService
    ) -> None:
        """Test valid mood assessment metric."""
        # Create a mock mood metric
        metric = Mock()
        metric.metric_type = Mock()
        metric.metric_type.value = "mood_assessment"
        metric.mental_health_data = Mock()  # Non-None mental health data

        assert health_data_service._validate_metric_business_rules(metric) is True

    def test_validate_business_rules_no_metric_type(
        self, health_data_service: HealthDataService
    ) -> None:
        """Test metric without type."""
        # Create a mock metric with no type
        metric = Mock()
        metric.metric_type = None

        assert health_data_service._validate_metric_business_rules(metric) is False

    def test_validate_business_rules_exception(
        self, health_data_service: HealthDataService
    ) -> None:
        """Test validation with exception."""
        metric = Mock()
        metric.metric_type = Mock(side_effect=AttributeError("Invalid"))

        assert health_data_service._validate_metric_business_rules(metric) is False


class TestExceptionClasses:
    """Test custom exception classes."""

    def test_health_data_service_error(self) -> None:
        """Test HealthDataServiceError."""
        error = HealthDataServiceError("Test error", status_code=400)
        assert str(error) == "Test error"
        assert error.status_code == 400

    def test_health_data_service_error_default_status(self) -> None:
        """Test HealthDataServiceError with default status."""
        error = HealthDataServiceError("Test error")
        assert error.status_code == 500

    def test_data_not_found_error(self) -> None:
        """Test DataNotFoundError."""
        error = DataNotFoundError("Data not found")
        assert str(error) == "Data not found"
        assert error.status_code == 404
        assert isinstance(error, HealthDataServiceError)

    def test_ml_prediction_error(self) -> None:
        """Test MLPredictionError."""
        error = MLPredictionError("Prediction failed", model_name="test_model")
        assert "ML Prediction Error in test_model" in str(error)
        assert error.status_code == 503
        assert error.model_name == "test_model"

    def test_ml_prediction_error_no_model(self) -> None:
        """Test MLPredictionError without model name."""
        error = MLPredictionError("Prediction failed")
        assert str(error) == "ML Prediction Error: Prediction failed"
        assert error.model_name is None
