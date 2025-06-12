"""Comprehensive tests for health data API endpoints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import uuid

from fastapi import status
from fastapi.testclient import TestClient
import pytest

from clarity.api.v1.health_data import (
    DependencyContainer,
    router,
    set_dependencies,
)
from clarity.auth.dependencies import AuthenticatedUser
from clarity.core.exceptions import (
    AuthorizationProblem,
    ResourceNotFoundProblem,
    ServiceUnavailableProblem,
    ValidationProblem,
)
from clarity.models.health_data import (
    HealthDataResponse,
    HealthDataUpload,
    HealthMetric,
    HealthMetricType,
    BiometricData,
    ProcessingStatus,
)
from clarity.ports.auth_ports import IAuthProvider
from clarity.ports.config_ports import IConfigProvider
from clarity.ports.data_ports import IHealthDataRepository
from clarity.services.health_data_service import (
    HealthDataService,
    HealthDataServiceError,
)


@pytest.fixture
def mock_auth_provider():
    """Mock authentication provider."""
    provider = Mock(spec=IAuthProvider)
    provider.verify_token = AsyncMock()
    return provider


@pytest.fixture
def mock_repository():
    """Mock health data repository."""
    repo = Mock(spec=IHealthDataRepository)
    repo.save_health_data = AsyncMock(return_value=True)
    repo.get_processing_status = AsyncMock()
    repo.get_user_health_data = AsyncMock()
    repo.delete_health_data = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def mock_config_provider():
    """Mock configuration provider."""
    provider = Mock(spec=IConfigProvider)
    provider.get = Mock()
    return provider


@pytest.fixture
def test_user():
    """Create test authenticated user."""
    return AuthenticatedUser(
        user_id=str(uuid.uuid4()),
        email="test@example.com",
        name="Test User",
        roles=["user"],
        permissions=["health_data:read", "health_data:write"],
    )


@pytest.fixture
def valid_health_data_upload(test_user):
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


@pytest.fixture
def mock_publisher():
    """Mock health data publisher."""
    publisher = Mock()
    publisher.publish_health_data_upload = AsyncMock(return_value="msg-123")
    return publisher


@pytest.fixture
async def setup_dependencies(
    mock_auth_provider,
    mock_repository,
    mock_config_provider,
):
    """Set up dependencies for testing."""
    set_dependencies(mock_auth_provider, mock_repository, mock_config_provider)
    yield
    # Reset after test
    container = DependencyContainer()
    container.auth_provider = None
    container.repository = None
    container.config_provider = None


@pytest.fixture
def client():
    """Create test client."""
    from fastapi import FastAPI
    
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/health-data")
    return TestClient(app)


class TestHealthCheckEndpoints:
    """Test health check endpoints."""

    def test_health_check_basic(self, client):
        """Test basic health check endpoint."""
        response = client.get("/api/v1/health-data/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "health-data-api"
        assert "timestamp" in data

    def test_health_check_detailed_with_dependencies(
        self, client, setup_dependencies
    ):
        """Test detailed health check with dependencies configured."""
        response = client.get("/api/v1/health-data/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == "connected"
        assert data["authentication"] == "available"
        assert "metrics" in data
        assert "version" in data

    def test_health_check_without_dependencies(self, client):
        """Test health check without dependencies."""
        # Clear dependencies
        container = DependencyContainer()
        set_dependencies(None, None, None)
        
        response = client.get("/api/v1/health-data/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["database"] == "not_configured"
        assert data["authentication"] == "not_configured"


class TestUploadHealthData:
    """Test health data upload endpoint."""

    @pytest.mark.asyncio
    async def test_upload_success(
        self,
        client,
        setup_dependencies,
        test_user,
        valid_health_data_upload,
        mock_repository,
        mock_publisher,
    ):
        """Test successful health data upload."""
        # Mock health data service
        processing_id = uuid.uuid4()
        with patch(
            "clarity.api.v1.health_data.get_health_data_service"
        ) as mock_get_service:
            mock_service = Mock(spec=HealthDataService)
            mock_service.process_health_data = AsyncMock(
                return_value=HealthDataResponse(
                    processing_id=processing_id,
                    status=ProcessingStatus.RECEIVED,
                    total_metrics=len(valid_health_data_upload.metrics),
                    message="Health data received successfully",
                )
            )
            mock_get_service.return_value = mock_service

            # Mock authentication
            with patch(
                "clarity.api.v1.router.get_current_user",
                return_value=test_user,
            ):
                # Mock publisher
                with patch(
                    "clarity.api.v1.health_data.get_publisher",
                    return_value=mock_publisher,
                ):
                    # Mock GCS
                    with patch("clarity.api.v1.health_data.storage"):
                        response = client.post(
                            "/api/v1/health-data/upload",
                            json=valid_health_data_upload.model_dump(mode="json"),
                            headers={"Authorization": "Bearer test-token"},
                        )

        assert response.status_code == 201
        data = response.json()
        assert data["processing_id"] == str(processing_id)
        assert data["status"] == ProcessingStatus.RECEIVED.value
        assert data["total_metrics"] == 2
        
        # Verify service was called
        mock_service.process_health_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_authorization_error(
        self,
        client,
        setup_dependencies,
        test_user,
        valid_health_data_upload,
    ):
        """Test upload with authorization error (wrong user)."""
        # Change user_id to different user
        valid_health_data_upload.user_id = uuid.uuid4()
        
        with patch("clarity.api.v1.router.get_current_user", return_value=test_user):
            response = client.post(
                "/api/v1/health-data/upload",
                json=valid_health_data_upload.model_dump(mode="json"),
                headers={"Authorization": "Bearer test-token"},
            )

        assert response.status_code == 403
        data = response.json()
        assert data["type"] == "https://docs.clarity.health/errors/authorization"
        assert "cannot upload data for another user" in data["detail"]

    @pytest.mark.asyncio
    async def test_upload_too_many_metrics(
        self,
        client,
        setup_dependencies,
        test_user,
    ):
        """Test upload with too many metrics."""
        # Create upload with too many metrics
        metrics = []
        for i in range(10001):  # Over the 10000 limit
            metrics.append(
                HealthMetric(
                    metric_id=uuid.uuid4(),
                    metric_type=HealthMetricType.HEART_RATE,
                    created_at=datetime.now(UTC),
                    device_id="device-123",
                    biometric_data=BiometricData(heart_rate=72),
                )
            )
        
        upload = HealthDataUpload(
            user_id=uuid.UUID(test_user.user_id),
            metrics=metrics,
            upload_source="test",
            client_timestamp=datetime.now(UTC),
        )
        
        with patch("clarity.api.v1.router.get_current_user", return_value=test_user):
            response = client.post(
                "/api/v1/health-data/upload",
                json=upload.model_dump(mode="json"),
                headers={"Authorization": "Bearer test-token"},
            )

        assert response.status_code == 400
        data = response.json()
        assert data["type"] == "https://docs.clarity.health/errors/validation"
        assert "Too many metrics" in data["detail"]

    @pytest.mark.asyncio
    async def test_upload_service_error(
        self,
        client,
        setup_dependencies,
        test_user,
        valid_health_data_upload,
    ):
        """Test upload with service error."""
        with patch(
            "clarity.api.v1.health_data.get_health_data_service"
        ) as mock_get_service:
            mock_service = Mock(spec=HealthDataService)
            mock_service.process_health_data = AsyncMock(
                side_effect=HealthDataServiceError("Processing failed")
            )
            mock_get_service.return_value = mock_service

            with patch("clarity.api.v1.router.get_current_user", return_value=test_user):
                response = client.post(
                    "/api/v1/health-data/upload",
                    json=valid_health_data_upload.model_dump(mode="json"),
                    headers={"Authorization": "Bearer test-token"},
                )

        assert response.status_code == 400
        data = response.json()
        assert data["type"] == "https://docs.clarity.health/errors/validation"
        assert "Processing failed" in data["detail"]

    @pytest.mark.asyncio
    async def test_upload_gcs_failure_continues(
        self,
        client,
        setup_dependencies,
        test_user,
        valid_health_data_upload,
        mock_publisher,
    ):
        """Test upload continues even if GCS save fails."""
        processing_id = uuid.uuid4()
        with patch(
            "clarity.api.v1.health_data.get_health_data_service"
        ) as mock_get_service:
            mock_service = Mock(spec=HealthDataService)
            mock_service.process_health_data = AsyncMock(
                return_value=HealthDataResponse(
                    processing_id=processing_id,
                    status=ProcessingStatus.RECEIVED,
                    total_metrics=2,
                    message="Success",
                )
            )
            mock_get_service.return_value = mock_service

            with patch("clarity.api.v1.router.get_current_user", return_value=test_user):
                with patch(
                    "clarity.api.v1.health_data.get_publisher",
                    return_value=mock_publisher,
                ):
                    # Mock GCS to fail
                    with patch(
                        "clarity.api.v1.health_data.storage.Client",
                        side_effect=Exception("GCS error"),
                    ):
                        response = client.post(
                            "/api/v1/health-data/upload",
                            json=valid_health_data_upload.model_dump(mode="json"),
                            headers={"Authorization": "Bearer test-token"},
                        )

        # Should still succeed
        assert response.status_code == 201
        data = response.json()
        assert data["processing_id"] == str(processing_id)


class TestProcessingStatus:
    """Test processing status endpoint."""

    @pytest.mark.asyncio
    async def test_get_processing_status_success(
        self,
        client,
        setup_dependencies,
        test_user,
    ):
        """Test successful status retrieval."""
        processing_id = uuid.uuid4()
        
        with patch(
            "clarity.api.v1.health_data.get_health_data_service"
        ) as mock_get_service:
            mock_service = Mock(spec=HealthDataService)
            mock_service.get_processing_status = AsyncMock(
                return_value={
                    "processing_id": str(processing_id),
                    "status": "completed",
                    "created_at": datetime.now(UTC).isoformat(),
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )
            mock_get_service.return_value = mock_service

            with patch("clarity.api.v1.router.get_current_user", return_value=test_user):
                response = client.get(
                    f"/api/v1/health-data/processing/{processing_id}",
                    headers={"Authorization": "Bearer test-token"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["processing_id"] == str(processing_id)
        assert data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_processing_status_not_found(
        self,
        client,
        setup_dependencies,
        test_user,
    ):
        """Test status retrieval when not found."""
        processing_id = uuid.uuid4()
        
        with patch(
            "clarity.api.v1.health_data.get_health_data_service"
        ) as mock_get_service:
            mock_service = Mock(spec=HealthDataService)
            mock_service.get_processing_status = AsyncMock(return_value=None)
            mock_get_service.return_value = mock_service

            with patch("clarity.api.v1.router.get_current_user", return_value=test_user):
                response = client.get(
                    f"/api/v1/health-data/processing/{processing_id}",
                    headers={"Authorization": "Bearer test-token"},
                )

        assert response.status_code == 404
        data = response.json()
        assert data["type"] == "https://docs.clarity.health/errors/resource-not-found"
        assert data["resource_type"] == "Processing Job"


class TestListHealthData:
    """Test list health data endpoint."""

    @pytest.mark.asyncio
    async def test_list_health_data_success(
        self,
        client,
        setup_dependencies,
        test_user,
    ):
        """Test successful health data listing."""
        with patch(
            "clarity.api.v1.health_data.get_health_data_service"
        ) as mock_get_service:
            mock_service = Mock(spec=HealthDataService)
            mock_service.get_user_health_data = AsyncMock(
                return_value={
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
            )
            mock_get_service.return_value = mock_service

            with patch("clarity.api.v1.router.get_current_user", return_value=test_user):
                response = client.get(
                    "/api/v1/health-data/",
                    headers={"Authorization": "Bearer test-token"},
                )

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2
        assert data["pagination"]["limit"] == 50
        assert data["pagination"]["has_next"] is False
        assert "links" in data

    @pytest.mark.asyncio
    async def test_list_health_data_with_filters(
        self,
        client,
        setup_dependencies,
        test_user,
    ):
        """Test health data listing with filters."""
        start_date = datetime.now(UTC) - timedelta(days=7)
        end_date = datetime.now(UTC)
        
        with patch(
            "clarity.api.v1.health_data.get_health_data_service"
        ) as mock_get_service:
            mock_service = Mock(spec=HealthDataService)
            mock_service.get_user_health_data = AsyncMock(
                return_value={"metrics": [], "pagination": {}}
            )
            mock_get_service.return_value = mock_service

            with patch("clarity.api.v1.router.get_current_user", return_value=test_user):
                response = client.get(
                    "/api/v1/health-data/",
                    params={
                        "data_type": "heart_rate",
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "source": "apple_watch",
                        "limit": 100,
                    },
                    headers={"Authorization": "Bearer test-token"},
                )

        assert response.status_code == 200
        
        # Verify service was called with correct params
        mock_service.get_user_health_data.assert_called_once()
        call_args = mock_service.get_user_health_data.call_args[1]
        assert call_args["metric_type"] == "heart_rate"
        assert call_args["limit"] == 100

    @pytest.mark.asyncio
    async def test_list_health_data_invalid_pagination(
        self,
        client,
        setup_dependencies,
        test_user,
    ):
        """Test listing with invalid pagination params."""
        with patch("clarity.api.v1.router.get_current_user", return_value=test_user):
            response = client.get(
                "/api/v1/health-data/",
                params={"limit": 0},  # Invalid limit
                headers={"Authorization": "Bearer test-token"},
            )

        assert response.status_code == 422  # FastAPI validation error


class TestDeleteHealthData:
    """Test delete health data endpoint."""

    @pytest.mark.asyncio
    async def test_delete_health_data_success(
        self,
        client,
        setup_dependencies,
        test_user,
    ):
        """Test successful health data deletion."""
        processing_id = uuid.uuid4()
        
        with patch(
            "clarity.api.v1.health_data.get_health_data_service"
        ) as mock_get_service:
            mock_service = Mock(spec=HealthDataService)
            mock_service.delete_health_data = AsyncMock(return_value=True)
            mock_get_service.return_value = mock_service

            with patch("clarity.api.v1.router.get_current_user", return_value=test_user):
                response = client.delete(
                    f"/api/v1/health-data/{processing_id}",
                    headers={"Authorization": "Bearer test-token"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Health data deleted successfully"
        assert data["processing_id"] == str(processing_id)
        assert "deleted_at" in data

    @pytest.mark.asyncio
    async def test_delete_health_data_not_found(
        self,
        client,
        setup_dependencies,
        test_user,
    ):
        """Test deletion when data not found."""
        processing_id = uuid.uuid4()
        
        with patch(
            "clarity.api.v1.health_data.get_health_data_service"
        ) as mock_get_service:
            mock_service = Mock(spec=HealthDataService)
            mock_service.delete_health_data = AsyncMock(return_value=False)
            mock_get_service.return_value = mock_service

            with patch("clarity.api.v1.router.get_current_user", return_value=test_user):
                response = client.delete(
                    f"/api/v1/health-data/{processing_id}",
                    headers={"Authorization": "Bearer test-token"},
                )

        assert response.status_code == 404
        data = response.json()
        assert data["type"] == "https://docs.clarity.health/errors/resource-not-found"


class TestLegacyEndpoints:
    """Test legacy endpoints."""

    def test_query_endpoint_removed(self, client):
        """Test that legacy query endpoint returns 410 Gone."""
        response = client.get("/api/v1/health-data/query")
        
        assert response.status_code == 410
        data = response.json()
        assert "Endpoint Permanently Removed" in data["detail"]["error"]
        assert "migration" in data["detail"]
        assert "new_endpoint" in data["detail"]["migration"]


class TestDependencyInjection:
    """Test dependency injection system."""

    def test_get_health_data_service_without_repository(self):
        """Test service getter without repository."""
        from clarity.api.v1.health_data import _container, get_health_data_service
        
        # Clear repository
        _container.repository = None
        
        with pytest.raises(ServiceUnavailableProblem) as exc_info:
            get_health_data_service()
        
        assert exc_info.value.service_name == "Health Data Repository"

    def test_get_auth_provider_without_provider(self):
        """Test auth provider getter without provider."""
        from clarity.api.v1.health_data import _container, get_auth_provider
        
        # Clear auth provider
        _container.auth_provider = None
        
        with pytest.raises(ServiceUnavailableProblem) as exc_info:
            get_auth_provider()
        
        assert exc_info.value.service_name == "Authentication Provider"

    def test_get_config_provider_without_provider(self):
        """Test config provider getter without provider."""
        from clarity.api.v1.health_data import _container, get_config_provider
        
        # Clear config provider
        _container.config_provider = None
        
        with pytest.raises(ServiceUnavailableProblem) as exc_info:
            get_config_provider()
        
        assert exc_info.value.service_name == "Configuration Provider"


class TestHelperFunctions:
    """Test helper functions."""

    def test_raise_authorization_error(self):
        """Test authorization error helper."""
        from clarity.api.v1.health_data import _raise_authorization_error
        
        with pytest.raises(AuthorizationProblem) as exc_info:
            _raise_authorization_error("user-123")
        
        assert "cannot upload data for another user" in str(exc_info.value.detail)

    def test_raise_not_found_error(self):
        """Test not found error helper."""
        from clarity.api.v1.health_data import _raise_not_found_error
        
        with pytest.raises(ResourceNotFoundProblem) as exc_info:
            _raise_not_found_error("Processing Job", "job-123")
        
        assert exc_info.value.resource_type == "Processing Job"
        assert exc_info.value.resource_id == "job-123"

    def test_raise_too_many_metrics_error(self):
        """Test too many metrics error helper."""
        from clarity.api.v1.health_data import _raise_too_many_metrics_error
        
        with pytest.raises(ValidationProblem) as exc_info:
            _raise_too_many_metrics_error(15000, 10000)
        
        assert "15000 exceeds maximum 10000" in str(exc_info.value.detail)