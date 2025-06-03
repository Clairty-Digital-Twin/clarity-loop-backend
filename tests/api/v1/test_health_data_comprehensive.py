"""Comprehensive tests for health_data API endpoints.

Tests all endpoints and edge cases to improve coverage from 33% to 85%+.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi import status
from fastapi.testclient import TestClient
import pytest

from clarity.api.v1.health_data import router
from clarity.models.health_data import BiometricData, HealthMetric, HealthMetricType


class TestHealthDataAPIComprehensive:
    """Comprehensive test coverage for health_data API endpoints."""

    @pytest.fixture
    def mock_health_service(self):
        """Create mock health data service."""
        service = AsyncMock()
        return service

    @pytest.fixture
    def mock_auth_service(self):
        """Create mock auth service."""
        service = AsyncMock()
        service.verify_token.return_value = {"user_id": "test_user", "email": "test@example.com"}
        return service

    @pytest.fixture
    def client_with_mocks(self, mock_health_service, mock_auth_service):
        """Create test client with mocked dependencies."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")

        # Mock the dependency injection
        with patch('clarity.api.v1.health_data.get_health_data_service', return_value=mock_health_service):
            with patch('clarity.api.v1.health_data.get_auth_service', return_value=mock_auth_service):
                with TestClient(app) as client:
                    yield client, mock_health_service, mock_auth_service

    @pytest.fixture
    def sample_health_data(self):
        """Create sample health data for testing."""
        return {
            "user_id": str(uuid4()),
            "metrics": [
                {
                    "metric_id": str(uuid4()),
                    "metric_type": "heart_rate",
                    "device_id": "test_device",
                    "raw_data": {},
                    "metadata": {},
                    "created_at": datetime.now(UTC).isoformat(),
                    "biometric_data": {
                        "heart_rate": 75.0,
                        "blood_pressure_systolic": 120,
                        "blood_pressure_diastolic": 80,
                        "oxygen_saturation": 99.0,
                        "heart_rate_variability": 50.0,
                        "respiratory_rate": 16.0,
                        "body_temperature": 37.0,
                        "blood_glucose": 100.0
                    }
                }
            ],
            "upload_source": "test_client",
            "client_timestamp": datetime.now(UTC).isoformat()
        }

    def test_get_health_data_success(self, client_with_mocks, sample_health_data):
        """Test successful health data retrieval."""
        client, mock_service, mock_auth = client_with_mocks

        # Mock successful response
        mock_service.get_user_health_data.return_value = {
            "data": [sample_health_data["metrics"][0]],
            "total_count": 1,
            "page_info": {
                "limit": 100,
                "offset": 0,
                "has_more": False
            }
        }

        response = client.get(
            "/api/v1/health-data",
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "data" in data
        assert "total_count" in data
        assert "page_info" in data

    def test_get_health_data_with_filters(self, client_with_mocks):
        """Test health data retrieval with filters."""
        client, mock_service, mock_auth = client_with_mocks

        mock_service.get_user_health_data.return_value = {
            "data": [],
            "total_count": 0,
            "page_info": {"limit": 50, "offset": 0, "has_more": False}
        }

        response = client.get(
            "/api/v1/health-data",
            params={
                "metric_type": "heart_rate",
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-12-31T23:59:59Z",
                "limit": 50,
                "offset": 0
            },
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == status.HTTP_200_OK
        mock_service.get_user_health_data.assert_called_once()

    def test_get_health_data_unauthorized(self, client_with_mocks):
        """Test health data retrieval without authorization."""
        client, mock_service, mock_auth = client_with_mocks

        # Mock auth failure
        mock_auth.verify_token.side_effect = Exception("Invalid token")

        response = client.get("/api/v1/health-data")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_health_data_service_error(self, client_with_mocks):
        """Test health data retrieval with service error."""
        client, mock_service, mock_auth = client_with_mocks

        # Mock service error
        mock_service.get_user_health_data.side_effect = Exception("Service error")

        response = client.get(
            "/api/v1/health-data",
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_upload_health_data_success(self, client_with_mocks, sample_health_data):
        """Test successful health data upload."""
        client, mock_service, mock_auth = client_with_mocks

        # Mock successful upload
        processing_id = str(uuid4())
        mock_service.upload_health_data.return_value = processing_id

        response = client.post(
            "/api/v1/health-data/upload",
            json=sample_health_data,
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert "processing_id" in data
        assert data["processing_id"] == processing_id

    def test_upload_health_data_invalid_payload(self, client_with_mocks):
        """Test health data upload with invalid payload."""
        client, mock_service, mock_auth = client_with_mocks

        invalid_data = {"invalid": "data"}

        response = client.post(
            "/api/v1/health-data/upload",
            json=invalid_data,
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_upload_health_data_empty_metrics(self, client_with_mocks, sample_health_data):
        """Test health data upload with empty metrics."""
        client, mock_service, mock_auth = client_with_mocks

        sample_health_data["metrics"] = []

        response = client.post(
            "/api/v1/health-data/upload",
            json=sample_health_data,
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_upload_health_data_service_error(self, client_with_mocks, sample_health_data):
        """Test health data upload with service error."""
        client, mock_service, mock_auth = client_with_mocks

        # Mock service error
        mock_service.upload_health_data.side_effect = Exception("Upload failed")

        response = client.post(
            "/api/v1/health-data/upload",
            json=sample_health_data,
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_get_processing_status_success(self, client_with_mocks):
        """Test successful processing status retrieval."""
        client, mock_service, mock_auth = client_with_mocks

        processing_id = str(uuid4())
        mock_service.get_processing_status.return_value = {
            "processing_id": processing_id,
            "status": "completed",
            "progress": 100,
            "created_at": datetime.now(UTC).isoformat()
        }

        response = client.get(
            f"/api/v1/health-data/status/{processing_id}",
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["processing_id"] == processing_id
        assert data["status"] == "completed"

    def test_get_processing_status_not_found(self, client_with_mocks):
        """Test processing status retrieval for non-existent job."""
        client, mock_service, mock_auth = client_with_mocks

        processing_id = str(uuid4())
        mock_service.get_processing_status.return_value = None

        response = client.get(
            f"/api/v1/health-data/status/{processing_id}",
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_processing_status_invalid_id(self, client_with_mocks):
        """Test processing status retrieval with invalid ID."""
        client, mock_service, mock_auth = client_with_mocks

        response = client.get(
            "/api/v1/health-data/status/invalid-id",
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_delete_health_data_success(self, client_with_mocks):
        """Test successful health data deletion."""
        client, mock_service, mock_auth = client_with_mocks

        mock_service.delete_health_data.return_value = True

        response = client.delete(
            "/api/v1/health-data",
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Health data deleted successfully"

    def test_delete_health_data_specific_processing(self, client_with_mocks):
        """Test deletion of specific processing job."""
        client, mock_service, mock_auth = client_with_mocks

        processing_id = str(uuid4())
        mock_service.delete_health_data.return_value = True

        response = client.delete(
            "/api/v1/health-data",
            params={"processing_id": processing_id},
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == status.HTTP_200_OK

    def test_delete_health_data_failure(self, client_with_mocks):
        """Test health data deletion failure."""
        client, mock_service, mock_auth = client_with_mocks

        mock_service.delete_health_data.return_value = False

        response = client.delete(
            "/api/v1/health-data",
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
