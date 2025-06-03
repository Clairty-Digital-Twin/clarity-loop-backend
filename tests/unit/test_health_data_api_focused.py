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
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from fastapi import status
import pytest
from starlette.datastructures import Headers

from clarity.api.v1.health_data import router as health_data_router
from clarity.core.exceptions import DataValidationError
from clarity.models.health_data import ActivityData, HealthDataUpload
from clarity.services.health_data_service import HealthDataServiceError
from tests.base import BaseServiceTestCase


class MockRequest:
    """Mock request object."""

    def __init__(self, user_id: str | None = None, headers: dict = None) -> None:
        """Initialize mock request."""
        self.state = Mock()
        self.state.user_id = user_id
        self.headers = Headers(headers or {})


class MockHealthDataService:
    """Mock health data service."""

    def __init__(self) -> None:
        """Initialize mock service."""
        self.should_fail = False
        self.fail_with = Exception("Service error")
        self.upload_result = "test_processing_id"

    async def upload_health_data(
        self, user_id: str, upload: HealthDataUpload
    ) -> str:
        """Mock upload health data."""
        if self.should_fail:
            raise self.fail_with
        return self.upload_result

    async def get_user_metrics(
        self, user_id: str, start_date: datetime = None, end_date: datetime = None
    ) -> dict:
        """Mock get user metrics."""
        if self.should_fail:
            raise self.fail_with
        return {
            "user_id": user_id,
            "metrics": [],
            "period": {"start": start_date, "end": end_date},
        }


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
        request = MockRequest(user_id=None)

        upload_data = HealthDataUpload(
            source="test_app",
            activity_data=[
                ActivityData(
                    steps=1000,
                    distance=1.0,
                    active_energy=100.0,
                    exercise_minutes=30,
                    flights_climbed=5,
                    active_minutes=30,
                    resting_heart_rate=65.0,
                )
            ],
        )

        # Act & Assert
        with pytest.raises(Exception):  # Should raise some form of auth error
            await health_data_router.upload_health_data(request, upload_data)

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_upload_health_data_service_validation_error(
        self, mock_get_service: Mock
    ) -> None:
        """Test upload with service validation error."""
        # Arrange
        self.mock_service.should_fail = True
        self.mock_service.fail_with = DataValidationError("Invalid data format")
        mock_get_service.return_value = self.mock_service

        request = MockRequest(user_id=str(uuid4()))
        upload_data = HealthDataUpload(
            source="test_app",
            activity_data=[
                ActivityData(
                    steps=1000,
                    distance=1.0,
                    active_energy=100.0,
                    exercise_minutes=30,
                    flights_climbed=5,
                    active_minutes=30,
                    resting_heart_rate=65.0,
                )
            ],
        )

        # Act & Assert
        with pytest.raises(DataValidationError):
            await health_data_router.upload_health_data(request, upload_data)

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_upload_health_data_service_error(
        self, mock_get_service: Mock
    ) -> None:
        """Test upload with general service error."""
        # Arrange
        self.mock_service.should_fail = True
        self.mock_service.fail_with = HealthDataServiceError("Database connection failed")
        mock_get_service.return_value = self.mock_service

        request = MockRequest(user_id=str(uuid4()))
        upload_data = HealthDataUpload(
            source="test_app",
            activity_data=[
                ActivityData(
                    steps=1000,
                    distance=1.0,
                    active_energy=100.0,
                    exercise_minutes=30,
                    flights_climbed=5,
                    active_minutes=30,
                    resting_heart_rate=65.0,
                )
            ],
        )

        # Act & Assert
        with pytest.raises(HealthDataServiceError):
            await health_data_router.upload_health_data(request, upload_data)

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_upload_health_data_unexpected_error(
        self, mock_get_service: Mock
    ) -> None:
        """Test upload with unexpected error."""
        # Arrange
        self.mock_service.should_fail = True
        self.mock_service.fail_with = RuntimeError("Unexpected system error")
        mock_get_service.return_value = self.mock_service

        request = MockRequest(user_id=str(uuid4()))
        upload_data = HealthDataUpload(
            source="test_app",
            activity_data=[
                ActivityData(
                    steps=1000,
                    distance=1.0,
                    active_energy=100.0,
                    exercise_minutes=30,
                    flights_climbed=5,
                    active_minutes=30,
                    resting_heart_rate=65.0,
                )
            ],
        )

        # Act & Assert
        with pytest.raises(RuntimeError):
            await health_data_router.upload_health_data(request, upload_data)


class TestHealthDataMetricsEndpoint(BaseServiceTestCase):
    """Test health data metrics endpoint - CHUNK 2B."""

    def setUp(self) -> None:
        """Set up test dependencies."""
        super().setUp()
        self.mock_service = MockHealthDataService()

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_get_metrics_missing_user_id(
        self, mock_get_service: Mock
    ) -> None:
        """Test metrics with missing user ID."""
        # Arrange
        mock_get_service.return_value = self.mock_service
        request = MockRequest(user_id=None)

        # Act & Assert
        with pytest.raises(Exception):  # Should raise auth error
            await health_data_router.get_user_metrics(request)

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_get_metrics_service_error(
        self, mock_get_service: Mock
    ) -> None:
        """Test metrics with service error."""
        # Arrange
        self.mock_service.should_fail = True
        self.mock_service.fail_with = HealthDataServiceError("Failed to fetch metrics")
        mock_get_service.return_value = self.mock_service

        request = MockRequest(user_id=str(uuid4()))

        # Act & Assert
        with pytest.raises(HealthDataServiceError):
            await health_data_router.get_user_metrics(request)

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_get_metrics_with_date_filters(
        self, mock_get_service: Mock
    ) -> None:
        """Test metrics with date filters."""
        # Arrange
        mock_get_service.return_value = self.mock_service
        request = MockRequest(user_id=str(uuid4()))

        start_date = datetime.now(UTC).replace(day=1)
        end_date = datetime.now(UTC)

        # Act
        result = await health_data_router.get_user_metrics(
            request, start_date=start_date, end_date=end_date
        )

        # Assert
        assert result["period"]["start"] == start_date
        assert result["period"]["end"] == end_date

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_get_metrics_invalid_date_range(
        self, mock_get_service: Mock
    ) -> None:
        """Test metrics with invalid date range."""
        # Arrange
        mock_get_service.return_value = self.mock_service
        request = MockRequest(user_id=str(uuid4()))

        # Invalid: end date before start date
        start_date = datetime.now(UTC)
        end_date = datetime.now(UTC).replace(day=1)

        # Act & Assert
        with pytest.raises(DataValidationError):
            await health_data_router.get_user_metrics(
                request, start_date=start_date, end_date=end_date
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
        request = MockRequest(user_id=None)

        # Act & Assert
        with pytest.raises(Exception):
            await health_data_router.query_health_data(request)

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_query_health_data_with_filters(
        self, mock_get_service: Mock
    ) -> None:
        """Test query with various filters."""
        # Arrange
        self.mock_service.query_result = {"data": [], "total": 0}
        mock_get_service.return_value = self.mock_service

        request = MockRequest(user_id=str(uuid4()))

        # Act
        result = await health_data_router.query_health_data(
            request,
            metric_type="activity",
            start_date=datetime.now(UTC).replace(day=1),
            end_date=datetime.now(UTC),
            limit=50,
            offset=0,
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
    async def test_empty_upload_data(
        self, mock_get_service: Mock
    ) -> None:
        """Test upload with empty data."""
        # Arrange
        mock_get_service.return_value = self.mock_service
        request = MockRequest(user_id=str(uuid4()))

        # Empty upload data
        upload_data = HealthDataUpload(source="test_app")

        # Act & Assert
        with pytest.raises(DataValidationError):
            await health_data_router.upload_health_data(request, upload_data)

    @patch("clarity.api.v1.health_data.get_health_data_service")
    async def test_invalid_source_format(
        self, mock_get_service: Mock
    ) -> None:
        """Test upload with invalid source format."""
        # Arrange
        mock_get_service.return_value = self.mock_service
        request = MockRequest(user_id=str(uuid4()))

        # Invalid source (empty string)
        upload_data = HealthDataUpload(
            source="",  # Invalid empty source
            activity_data=[
                ActivityData(
                    steps=1000,
                    distance=1.0,
                    active_energy=100.0,
                    exercise_minutes=30,
                    flights_climbed=5,
                    active_minutes=30,
                    resting_heart_rate=65.0,
                )
            ],
        )

        # Act & Assert
        with pytest.raises(DataValidationError):
            await health_data_router.upload_health_data(request, upload_data)

    async def test_request_state_validation(self) -> None:
        """Test request state validation utility."""
        # This tests the internal validation logic
        from clarity.api.v1.health_data import _validate_request_state

        # Valid request
        valid_request = MockRequest(user_id=str(uuid4()))
        user_id = _validate_request_state(valid_request)
        assert user_id is not None

        # Invalid request
        invalid_request = MockRequest(user_id=None)
        with pytest.raises(Exception):
            _validate_request_state(invalid_request)

    async def test_date_range_validation(self) -> None:
        """Test date range validation utility."""
        from clarity.api.v1.health_data import _validate_date_range

        # Valid range
        start = datetime.now(UTC).replace(day=1)
        end = datetime.now(UTC)
        _validate_date_range(start, end)  # Should not raise

        # Invalid range
        with pytest.raises(DataValidationError):
            _validate_date_range(end, start)  # end before start

    async def test_upload_data_validation(self) -> None:
        """Test upload data validation utility."""
        from clarity.api.v1.health_data import _validate_upload_data

        # Valid data
        valid_upload = HealthDataUpload(
            source="test_app",
            activity_data=[
                ActivityData(
                    steps=1000,
                    distance=1.0,
                    active_energy=100.0,
                    exercise_minutes=30,
                    flights_climbed=5,
                    active_minutes=30,
                    resting_heart_rate=65.0,
                )
            ],
        )
        _validate_upload_data(valid_upload)  # Should not raise

        # Invalid data - empty source
        invalid_upload = HealthDataUpload(source="")
        with pytest.raises(DataValidationError):
            _validate_upload_data(invalid_upload)

        # Invalid data - no data fields
        empty_upload = HealthDataUpload(source="test_app")
        with pytest.raises(DataValidationError):
            _validate_upload_data(empty_upload)
