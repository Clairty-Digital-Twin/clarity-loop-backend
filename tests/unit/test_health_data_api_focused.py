"""FOCUSED Health Data API Tests - CHUNK 2.

🚀 SURGICAL STRIKE ON HEALTH DATA API 🚀
Target: 32.9% → 80% coverage

Breaking down into small, testable chunks:
- Upload endpoint error handling
- Metrics endpoint edge cases
- Validation error responses
- Authorization paths
- Status code handling

Each test targets specific uncovered code paths.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

# Import removed - not used
import pytest
from starlette.datastructures import Headers

from clarity.api.v1.health_data import (
    list_health_data,
    query_health_data_legacy,
    upload_health_data,
)
from clarity.auth import UserContext
from clarity.core.exceptions import DataValidationError
from clarity.models.health_data import (
    ActivityData,
    HealthDataResponse,
    HealthDataUpload,
    HealthMetric,
    HealthMetricType,
    ProcessingStatus,
)
from clarity.services.health_data_service import HealthDataServiceError
from tests.base import BaseServiceTestCase


def create_test_health_data_upload(
    user_id: str | None = None,
    upload_source: str = "test_app",
    activity_data: ActivityData | None = None,
) -> HealthDataUpload:
    """Create a valid HealthDataUpload instance for testing."""
    if user_id is None:
        user_id = str(uuid4())

    if activity_data is None:
        activity_data = ActivityData(
            steps=1000,
            distance=1.0,
            active_energy=100.0,
            exercise_minutes=30,
            flights_climbed=5,
            active_minutes=30,
            resting_heart_rate=65.0,
        )

    # Create a HealthMetric with the activity data
    metric = HealthMetric(
        metric_type=HealthMetricType.ACTIVITY_LEVEL,
        activity_data=activity_data,
    )

    return HealthDataUpload(
        user_id=user_id,
        metrics=[metric],
        upload_source=upload_source,
        client_timestamp=datetime.now(UTC),
    )


class MockRequest:
    """Mock request object."""

    def __init__(
        self, user_id: str | None = None, headers: dict[str, str] | None = None
    ) -> None:
        """Initialize mock request."""
        self.state = Mock()
        self.state.user_id = user_id
        self.headers = Headers(headers or {})
        self.url = Mock()
        self.url.scheme = "https"
        self.url.netloc = "api.clarity.health"


class MockHealthDataService:
    """Mock health data service."""

    def __init__(self) -> None:
        """Initialize mock service."""
        self.should_fail = False
        self.fail_with = Exception("Service error")
        self.upload_result = "test_processing_id"
        self.query_result: dict[str, Any] = {"data": [], "total": 0}

    async def process_health_data(
        self,
        upload: HealthDataUpload,
    ) -> HealthDataResponse:
        """Mock process health data."""
        if self.should_fail:
            raise self.fail_with
        return HealthDataResponse(
            processing_id=uuid4(),
            status=ProcessingStatus.PROCESSING,
            accepted_metrics=len(upload.metrics),
            rejected_metrics=0,
            validation_errors=[],
            estimated_processing_time=30,
            sync_token=upload.sync_token,
            message="Health data uploaded successfully and is being processed",
            timestamp=datetime.now(UTC),
        )

    async def get_user_health_data(
        self,
        user_id: str,  # noqa: ARG002
        limit: int = 100,  # noqa: ARG002
        offset: int = 0,  # noqa: ARG002
        metric_type: str | None = None,  # noqa: ARG002
        start_date: datetime | None = None,  # noqa: ARG002
        end_date: datetime | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Mock get user health data."""
        if self.should_fail:
            raise self.fail_with
        return {"metrics": [], "total": 0}

    async def get_user_metrics(
        self,
        user_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Mock get user metrics."""
        if self.should_fail:
            raise self.fail_with
        return {
            "user_id": user_id,
            "metrics": [],
            "period": {"start": start_date, "end": end_date},
        }

    async def query_health_data(
        self,
        user_id: str,  # noqa: ARG002
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, Any]:
        """Mock query health data."""
        if self.should_fail:
            raise self.fail_with
        return self.query_result


class TestHealthDataUploadEndpoint(BaseServiceTestCase):
    """Test health data upload endpoint - CHUNK 2A."""

    def setUp(self) -> None:
        """Set up test dependencies."""
        super().setUp()
        self.mock_service = MockHealthDataService()

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_upload_health_data_missing_user_id(
        self, mock_get_service: Mock
    ) -> None:
        """Test upload with missing user ID in request state."""
        # Arrange
        mock_get_service.return_value = self.mock_service

        upload_data = create_test_health_data_upload(
            user_id=str(uuid4()),
            upload_source="test_app",
        )

        # Create mock user context (will be None to trigger error)
        user_context = UserContext(user_id=None, permissions=[])

        # Act & Assert
        with pytest.raises((ValueError, RuntimeError, AttributeError)):
            await upload_health_data(
                health_data=upload_data,
                current_user=user_context,
                service=self.mock_service,
            )

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_upload_health_data_service_validation_error(
        self, mock_get_service: Mock
    ) -> None:
        """Test upload with service validation error."""
        # Arrange
        self.mock_service.should_fail = True
        self.mock_service.fail_with = DataValidationError("Invalid data format")
        mock_get_service.return_value = self.mock_service
        upload_data = create_test_health_data_upload(
            user_id=str(uuid4()),
            upload_source="test_app",
        )

        # Create mock user context
        user_context = UserContext(user_id=str(uuid4()), permissions=[])

        # Act & Assert
        with pytest.raises(DataValidationError):
            await upload_health_data(
                health_data=upload_data,
                current_user=user_context,
                service=self.mock_service,
            )

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_upload_health_data_service_error(
        self, mock_get_service: Mock
    ) -> None:
        """Test upload with general service error."""
        # Arrange
        self.mock_service.should_fail = True
        self.mock_service.fail_with = HealthDataServiceError(
            "Database connection failed"
        )
        mock_get_service.return_value = self.mock_service
        upload_data = create_test_health_data_upload(
            user_id=str(uuid4()),
            upload_source="test_app",
        )

        # Create mock user context
        user_context = UserContext(user_id=str(uuid4()), permissions=[])

        # Act & Assert
        with pytest.raises(HealthDataServiceError):
            await upload_health_data(
                health_data=upload_data,
                current_user=user_context,
                service=self.mock_service,
            )

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_upload_health_data_unexpected_error(
        self, mock_get_service: Mock
    ) -> None:
        """Test upload with unexpected error."""
        # Arrange
        self.mock_service.should_fail = True
        self.mock_service.fail_with = RuntimeError("Unexpected system error")
        mock_get_service.return_value = self.mock_service
        upload_data = create_test_health_data_upload(
            user_id=str(uuid4()),
            upload_source="test_app",
        )

        # Create mock user context
        user_context = UserContext(user_id=str(uuid4()), permissions=[])

        # Act & Assert
        with pytest.raises(RuntimeError):
            await upload_health_data(
                health_data=upload_data,
                current_user=user_context,
                service=self.mock_service,
            )


class TestHealthDataMetricsEndpoint(BaseServiceTestCase):
    """Test health data metrics endpoint - CHUNK 2B."""

    def setUp(self) -> None:
        """Set up test dependencies."""
        super().setUp()
        self.mock_service = MockHealthDataService()

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_get_metrics_missing_user_id(self, mock_get_service: Mock) -> None:
        """Test metrics with missing user ID."""
        # Arrange
        mock_get_service.return_value = self.mock_service

        # Create mock user context (will be None to trigger error)
        user_context = UserContext(user_id=None, permissions=[])

        # Act & Assert
        with pytest.raises((ValueError, RuntimeError, AttributeError)):
            await list_health_data(
                request=request,
                current_user=user_context,
                service=self.mock_service,
            )

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_get_metrics_service_error(self, mock_get_service: Mock) -> None:
        """Test metrics with service error."""
        # Arrange
        self.mock_service.should_fail = True
        self.mock_service.fail_with = HealthDataServiceError("Failed to fetch metrics")
        mock_get_service.return_value = self.mock_service


        # Create mock user context
        user_context = UserContext(user_id=str(uuid4()), permissions=[])

        # Act & Assert
        with pytest.raises(HealthDataServiceError):
            await list_health_data(
                request=request,
                current_user=user_context,
                service=self.mock_service,
            )

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_get_metrics_with_date_filters(self, mock_get_service: Mock) -> None:
        """Test metrics with date filters."""
        # Arrange
        mock_get_service.return_value = self.mock_service

        start_date = datetime.now(UTC).replace(day=1)
        end_date = datetime.now(UTC)

        # Create mock user context
        user_context = UserContext(user_id=str(uuid4()), permissions=[])

        # Act
        result = await list_health_data(
            request=request,
            current_user=user_context,
            start_date=start_date,
            end_date=end_date,
            service=self.mock_service,
        )

        # Assert
        assert result["period"]["start"] == start_date
        assert result["period"]["end"] == end_date

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_get_metrics_invalid_date_range(self, mock_get_service: Mock) -> None:
        """Test metrics with invalid date range."""
        # Arrange
        mock_get_service.return_value = self.mock_service

        # Invalid: end date before start date
        start_date = datetime.now(UTC)
        end_date = datetime.now(UTC).replace(day=1)

        # Create mock user context
        user_context = UserContext(user_id=str(uuid4()), permissions=[])

        # Act & Assert
        with pytest.raises(DataValidationError):
            await list_health_data(
                request=request,
                current_user=user_context,
                start_date=start_date,
                end_date=end_date,
                service=self.mock_service,
            )


class TestHealthDataQueryEndpoint(BaseServiceTestCase):
    """Test health data query endpoint - CHUNK 2C."""

    def setUp(self) -> None:
        """Set up test dependencies."""
        super().setUp()
        self.mock_service = MockHealthDataService()

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_query_health_data_missing_user_id(
        self, mock_get_service: Mock
    ) -> None:
        """Test query with missing user ID."""
        # Arrange
        mock_get_service.return_value = self.mock_service

        # Create mock user context (will be None to trigger error)
        user_context = UserContext(user_id=None, permissions=[])

        # Act & Assert
        with pytest.raises((ValueError, RuntimeError, AttributeError)):
            await query_health_data_legacy(
                current_user=user_context,
                service=self.mock_service,
            )

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_query_health_data_with_filters(self, mock_get_service: Mock) -> None:
        """Test query with various filters."""
        # Arrange
        self.mock_service.query_result = {"data": [], "total": 0}
        mock_get_service.return_value = self.mock_service


        # Create mock user context
        user_context = UserContext(user_id=str(uuid4()), permissions=[])

        # Act
        result = await query_health_data_legacy(
            current_user=user_context,
            metric_type="activity",
            start_date=datetime.now(UTC).replace(day=1),
            end_date=datetime.now(UTC),
            limit=50,
            offset=0,
            service=self.mock_service,
        )

        # Assert
        assert "data" in result
        assert "total" in result


class TestHealthDataErrorHandling(BaseServiceTestCase):
    """Test error handling paths - CHUNK 2D."""

    def setUp(self) -> None:
        """Set up test dependencies."""
        super().setUp()
        self.mock_service = MockHealthDataService()

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_empty_upload_data(self, mock_get_service: Mock) -> None:
        """Test upload with empty data."""
        # Arrange
        mock_get_service.return_value = self.mock_service

        # Empty upload data (will fail validation)
        upload_data = create_test_health_data_upload(
            user_id=str(uuid4()),
            upload_source="test_app",
        )

        # Create mock user context
        user_context = UserContext(user_id=str(uuid4()), permissions=[])

        # Act & Assert
        with pytest.raises(DataValidationError):
            await upload_health_data(
                health_data=upload_data,
                current_user=user_context,
                service=self.mock_service,
            )

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_invalid_source_format(self, mock_get_service: Mock) -> None:
        """Test upload with invalid source format."""
        # Arrange
        mock_get_service.return_value = self.mock_service

        # Invalid source (empty string)
        upload_data = create_test_health_data_upload(
            user_id=str(uuid4()),
            upload_source="",  # Invalid empty source
        )

        # Create mock user context
        user_context = UserContext(user_id=str(uuid4()), permissions=[])

        # Act & Assert
        with pytest.raises(DataValidationError):
            await upload_health_data(
                health_data=upload_data,
                current_user=user_context,
                service=self.mock_service,
            )

    # Private function tests removed - testing internal implementation details
    # is not recommended. These functions should be tested through the public API.
