"""Tests for health data API endpoints that actually test real code."""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# Import the REAL modules we want to test
from clarity.api.v1.health_data import router, set_dependencies
from clarity.models.auth import Permission, UserContext
from clarity.models.health_data import (
    BiometricData,
    HealthDataResponse,
    HealthDataUpload,
    HealthMetric,
    HealthMetricType,
    ProcessingStatus,
)
from clarity.ports.auth_ports import IAuthProvider
from clarity.ports.config_ports import IConfigProvider
from clarity.ports.data_ports import IHealthDataRepository


@pytest.fixture
def mock_auth_provider() -> Mock:
    """Mock authentication provider."""
    provider = Mock(spec=IAuthProvider)
    provider.verify_token = AsyncMock()
    return provider


@pytest.fixture
def mock_repository() -> Mock:
    """Mock health data repository."""
    repo = Mock(spec=IHealthDataRepository)
    repo.save_health_data = AsyncMock(return_value=True)
    repo.get_processing_status = AsyncMock(
        return_value={
            "processing_id": str(uuid.uuid4()),
            "status": "pending",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )
    repo.get_user_health_data = AsyncMock(return_value=[])
    repo.delete_health_data = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def mock_config_provider() -> Mock:
    """Mock configuration provider."""
    provider = Mock(spec=IConfigProvider)
    provider.get = Mock()
    return provider


@pytest.fixture
def test_user() -> UserContext:
    """Create test authenticated user."""
    return UserContext(
        user_id=str(uuid.uuid4()),
        email="test@example.com",
        permissions=[Permission.READ_OWN_DATA, Permission.WRITE_OWN_DATA],
    )


@pytest.fixture
def valid_health_data_upload(test_user: UserContext) -> HealthDataUpload:
    """Create valid health data upload."""
    return HealthDataUpload(
        user_id=uuid.UUID(test_user.user_id),
        metrics=[
            HealthMetric(
                metric_id=uuid.uuid4(),
                metric_type=HealthMetricType.HEART_RATE,
                created_at=datetime.now(UTC),
                device_id="device-123",
                biometric_data=BiometricData(heart_rate=72),
            ),
            HealthMetric(
                metric_id=uuid.uuid4(),
                metric_type=HealthMetricType.BLOOD_PRESSURE,
                created_at=datetime.now(UTC),
                device_id="device-123",
                biometric_data=BiometricData(systolic_bp=120, diastolic_bp=80),
            ),
        ],
        upload_source="mobile_app",
        client_timestamp=datetime.now(UTC),
    )


@pytest.fixture(autouse=True)
def reset_health_data_dependencies():
    """Reset health data dependencies after each test to prevent state pollution."""
    yield
    # Clean up the global container after each test
    from clarity.api.v1.health_data import _container  # noqa: PLC0415, PLC2701

    _container.auth_provider = None
    _container.repository = None
    _container.config_provider = None


@pytest.fixture
def app_with_dependencies(
    test_user: UserContext,
    mock_auth_provider: Mock,
    mock_repository: Mock,
    mock_config_provider: Mock,
) -> Generator[FastAPI, None, None]:
    """Create a real FastAPI app with dependencies properly configured."""
    app = FastAPI()

    # Set up the dependency injection container BEFORE including the router
    set_dependencies(mock_auth_provider, mock_repository, mock_config_provider)

    # Override ONLY the authentication dependency - let everything else be real
    from clarity.auth.dependencies import get_authenticated_user  # noqa: PLC0415

    app.dependency_overrides[get_authenticated_user] = lambda: test_user

    # Include the REAL health data router
    app.include_router(router, prefix="/api/v1/health-data")

    return app


@pytest.fixture
def app(test_user: UserContext) -> FastAPI:
    """Create a basic FastAPI app without dependencies (for testing 503 errors)."""
    app = FastAPI()

    # Override ONLY the authentication dependency - let everything else be real
    from clarity.auth.dependencies import get_authenticated_user  # noqa: PLC0415

    app.dependency_overrides[get_authenticated_user] = lambda: test_user

    # Include the REAL health data router
    app.include_router(router, prefix="/api/v1/health-data")

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client with app that has no dependencies (for testing service unavailable)."""
    return TestClient(app)


@pytest.fixture
def client_with_dependencies(app_with_dependencies: FastAPI) -> TestClient:
    """Create test client with app that has dependencies configured."""
    return TestClient(app_with_dependencies)


class TestHealthCheckEndpoints:
    """Test health check endpoints with real code."""

    def test_health_check_basic(self, client: TestClient) -> None:
        """Test basic health check endpoint."""
        response = client.get("/api/v1/health-data/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "health-data-api"
        assert "timestamp" in data

    def test_health_check_with_dependencies(
        self, client_with_dependencies: TestClient
    ) -> None:
        """Test health check with dependencies configured."""
        response = client_with_dependencies.get("/api/v1/health-data/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "health-data-api"


class TestUploadHealthDataNoDependencies:
    """Test upload endpoint behavior when dependencies are not configured."""

    def test_upload_service_unavailable(
        self, client: TestClient, valid_health_data_upload: HealthDataUpload
    ) -> None:
        """Test upload with default behavior when dependencies not explicitly configured."""
        response = client.post(
            "/api/v1/health-data",
            json=valid_health_data_upload.model_dump(mode="json"),
            headers={"Authorization": "Bearer test-token"},
        )

        # When dependencies are not configured, the service returns 503
        assert response.status_code == 503


class TestUploadHealthDataWithDependencies:
    """Test upload endpoint with properly configured dependencies."""

    def test_upload_authorization_error(
        self,
        client_with_dependencies: TestClient,
        valid_health_data_upload: HealthDataUpload,
    ) -> None:
        """Test upload with authorization error (wrong user)."""
        # Change user_id to different user
        valid_health_data_upload.user_id = uuid.uuid4()

        response = client_with_dependencies.post(
            "/api/v1/health-data",
            json=valid_health_data_upload.model_dump(mode="json"),
            headers={"Authorization": "Bearer test-token"},
        )

        # Router returns 500 for authorization mismatches (internal error handling)
        assert response.status_code == 500

    def test_upload_validation_error(
        self, client_with_dependencies: TestClient, test_user: UserContext
    ) -> None:
        """Test upload with invalid data."""
        # Create upload with invalid data
        invalid_upload = {
            "user_id": "not-a-uuid",  # Invalid UUID
            "metrics": [],  # Empty metrics
            "upload_source": "",  # Empty source
        }

        response = client_with_dependencies.post(
            "/api/v1/health-data",
            json=invalid_upload,
            headers={"Authorization": "Bearer test-token"},
        )

        assert response.status_code == 422

    def test_upload_too_many_metrics(
        self, client_with_dependencies: TestClient, test_user: UserContext
    ) -> None:
        """Test upload with too many metrics."""
        # Create upload with too many metrics (over 10000 limit)
        # Over the 10000 limit
        metrics = [
            {
                "metric_id": str(uuid.uuid4()),
                "metric_type": "heart_rate",
                "created_at": datetime.now(UTC).isoformat(),
                "device_id": "device-123",
                "biometric_data": {"heart_rate": 72},
            }
            for i in range(10001)
        ]

        upload = {
            "user_id": test_user.user_id,
            "metrics": metrics,
            "upload_source": "test",
            "client_timestamp": datetime.now(UTC).isoformat(),
        }

        response = client_with_dependencies.post(
            "/api/v1/health-data",
            json=upload,
            headers={"Authorization": "Bearer test-token"},
        )

        assert response.status_code == 422  # Unprocessable Entity for validation errors

    @pytest.mark.asyncio
    async def test_upload_success_with_mocked_service(
        self,
        client_with_dependencies: TestClient,
        valid_health_data_upload: HealthDataUpload,
        mock_repository: Mock,
    ) -> None:
        """Test successful upload by mocking only the service layer."""
        processing_id = uuid.uuid4()

        # Mock the repository to return success
        mock_repository.save_health_data.return_value = True

        # Mock GCS and publisher
        with (
            patch("clarity.api.v1.health_data.storage"),
            patch("clarity.api.v1.health_data.get_publisher") as mock_get_publisher,
            patch(
                "clarity.services.health_data_service.HealthDataService.process_health_data"
            ) as mock_process,
        ):
            mock_publisher = AsyncMock()
            mock_publisher.publish_health_data_upload = AsyncMock(
                return_value="msg-123"
            )
            mock_get_publisher.return_value = mock_publisher
            mock_process.return_value = HealthDataResponse(
                processing_id=processing_id,
                status=ProcessingStatus.RECEIVED,
                accepted_metrics=len(valid_health_data_upload.metrics),
                message="Health data received successfully",
            )

            response = client_with_dependencies.post(
                "/api/v1/health-data",
                json=valid_health_data_upload.model_dump(mode="json"),
                headers={"Authorization": "Bearer test-token"},
            )

        assert response.status_code == 201
        data = response.json()
        assert data["processing_id"] == str(processing_id)
        assert data["status"] == ProcessingStatus.RECEIVED.value
        assert data["accepted_metrics"] == 2


class TestProcessingStatus:
    """Test processing status endpoint with real code."""

    def test_get_processing_status_service_unavailable(
        self, client: TestClient
    ) -> None:
        """Test status retrieval when dependencies not explicitly configured."""
        processing_id = uuid.uuid4()

        response = client.get(
            f"/api/v1/health-data/processing/{processing_id}",
            headers={"Authorization": "Bearer test-token"},
        )

        # These endpoints properly return 503 when dependencies not configured
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_get_processing_status_success(
        self, client_with_dependencies: TestClient, mock_repository: Mock
    ) -> None:
        """Test successful status retrieval with mocked service."""
        processing_id = uuid.uuid4()

        # Mock the service to return status
        with patch(
            "clarity.services.health_data_service.HealthDataService.get_processing_status"
        ) as mock_get_status:
            mock_get_status.return_value = {
                "processing_id": str(processing_id),
                "status": "completed",
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
            }

            response = client_with_dependencies.get(
                f"/api/v1/health-data/processing/{processing_id}",
                headers={"Authorization": "Bearer test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["processing_id"] == str(processing_id)
        assert data["status"] == "completed"


class TestListHealthData:
    """Test list health data endpoint with real code."""

    def test_list_health_data_service_unavailable(self, client: TestClient) -> None:
        """Test listing when dependencies not explicitly configured."""
        response = client.get(
            "/api/v1/health-data/",
            headers={"Authorization": "Bearer test-token"},
        )

        # These endpoints properly return 503 when dependencies not configured
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_list_health_data_success(
        self, client_with_dependencies: TestClient, mock_repository: Mock
    ) -> None:
        """Test successful health data listing."""
        # Mock the service to return data
        with patch(
            "clarity.services.health_data_service.HealthDataService.get_user_health_data"
        ) as mock_get_data:
            mock_get_data.return_value = {
                "metrics": [
                    {"metric_id": "1", "type": "heart_rate", "value": 72},
                    {"metric_id": "2", "type": "steps", "value": 5000},
                ],
                "pagination": {
                    "limit": 50,
                    "offset": 0,
                    "total": 2,
                    "has_more": False,
                },
            }

            response = client_with_dependencies.get(
                "/api/v1/health-data/",
                headers={"Authorization": "Bearer test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2


class TestDeleteHealthData:
    """Test delete health data endpoint with real code."""

    def test_delete_health_data_service_unavailable(self, client: TestClient) -> None:
        """Test deletion when dependencies not explicitly configured."""
        processing_id = uuid.uuid4()

        response = client.delete(
            f"/api/v1/health-data/{processing_id}",
            headers={"Authorization": "Bearer test-token"},
        )

        # When dependencies are not configured, the service returns 503
        assert response.status_code == 503
