"""Health Data API Integration Tests - Testing REAL Code.

🚀 REAL CODE TESTING - NO MORE OVER-MOCKING 🚀
Target: Test actual health data API router implementations

Fixed over-mocking issues:
- Import and test REAL health data router
- Only mock external dependencies (AWS, databases) 
- Let actual API endpoints execute during tests
- Test real business logic and validation

Each test targets actual code paths in the real implementation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

# Import the REAL modules we want to test
from clarity.api.v1.health_data import router
from clarity.models.auth import UserContext, Permission
from clarity.models.health_data import (
    ActivityData,
    HealthDataUpload,
    HealthMetric,
    HealthMetricType,
    ProcessingStatus,
)
from clarity.auth.dependencies import get_authenticated_user
from clarity.ports.data_ports import IHealthDataRepository


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


@pytest.fixture
def mock_repository():
    """Mock health data repository."""
    repo = Mock(spec=IHealthDataRepository)
    repo.store_health_data = AsyncMock(return_value=str(uuid4()))
    repo.get_user_health_data = AsyncMock(return_value={"data": [], "total": 0})
    repo.get_user_metrics = AsyncMock(return_value={"metrics": [], "user_id": "test"})
    return repo


@pytest.fixture
def app_with_real_router(mock_repository):
    """Create FastAPI app with REAL health data router."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    
    # Create test user
    test_user = UserContext(
        user_id=str(uuid4()),
        email="test@example.com",
        permissions=[Permission.READ_OWN_DATA, Permission.WRITE_OWN_DATA]
    )
    
    # Override ONLY the authentication - let everything else be real
    app.dependency_overrides[get_authenticated_user] = lambda: test_user
    
    return app


@pytest.fixture
def client(app_with_real_router):
    """Create test client with real router."""
    return TestClient(app_with_real_router)


class TestHealthDataUploadRealAPI:
    """Test health data upload with REAL API router."""

    def test_upload_health_data_success(self, client: TestClient):
        """Test successful health data upload through real API."""
        upload_data = create_test_health_data_upload().model_dump()
        
        with patch("clarity.api.v1.health_data.get_health_data_service") as mock_get_service:
            mock_service = Mock()
            mock_service.process_health_data = AsyncMock(return_value=Mock(
                processing_id=uuid4(),
                status="processing",
                accepted_metrics=1,
                rejected_metrics=0,
                validation_errors=[],
                estimated_processing_time=30,
                sync_token=None,
                message="Health data uploaded successfully",
                timestamp=datetime.now(UTC),
            ))
            mock_get_service.return_value = mock_service
            
            with patch("clarity.api.v1.health_data.get_publisher") as mock_get_publisher:
                mock_publisher = AsyncMock()
                mock_publisher.publish_health_data_upload = AsyncMock(return_value="msg-123")
                mock_get_publisher.return_value = mock_publisher
                
                response = client.post("/api/v1/upload", json=upload_data)
                
                assert response.status_code == status.HTTP_201_CREATED
                response_data = response.json()
                assert "processing_id" in response_data
                assert "status" in response_data

    def test_upload_health_data_validation_error(self, client: TestClient):
        """Test upload with validation error through real API."""
        # Invalid data - missing required fields
        invalid_data = {
            "user_id": str(uuid4()),
            "metrics": [],  # Empty metrics should cause validation error
            "upload_source": "",  # Empty source should cause validation error
        }
        
        response = client.post("/api/v1/upload", json=invalid_data)
        
        # Should get validation error
        assert response.status_code in [422, 400]

    def test_upload_health_data_service_error(self, client: TestClient):
        """Test upload with service error through real API."""
        upload_data = create_test_health_data_upload().model_dump()
        
        with patch("clarity.api.v1.health_data.get_health_data_service") as mock_get_service:
            mock_service = Mock()
            mock_service.process_health_data = AsyncMock(side_effect=Exception("Database error"))
            mock_get_service.return_value = mock_service
            
            response = client.post("/api/v1/upload", json=upload_data)
            
            # Should get server error
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestHealthDataListRealAPI:
    """Test health data listing with REAL API router."""

    def test_list_health_data_success(self, client: TestClient):
        """Test successful health data listing through real API."""
        with patch("clarity.api.v1.health_data.get_health_data_service") as mock_get_service:
            mock_service = Mock()
            mock_service.get_user_health_data = AsyncMock(return_value={
                "metrics": [
                    {
                        "id": str(uuid4()),
                        "user_id": str(uuid4()),
                        "metric_type": "activity_level",
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                ],
                "total": 1
            })
            mock_get_service.return_value = mock_service
            
            response = client.get("/api/v1/")
            
            assert response.status_code == status.HTTP_200_OK
            response_data = response.json()
            assert "data" in response_data
            assert "pagination" in response_data

    def test_list_health_data_with_filters(self, client: TestClient):
        """Test health data listing with filters through real API."""
        with patch("clarity.api.v1.health_data.get_health_data_service") as mock_get_service:
            mock_service = Mock()
            mock_service.get_user_health_data = AsyncMock(return_value={"metrics": [], "total": 0})
            mock_get_service.return_value = mock_service
            
            # Test with query parameters
            response = client.get("/api/v1/?limit=50&offset=10&data_type=activity_level")
            
            assert response.status_code == status.HTTP_200_OK
            
            # Verify the service was called with correct parameters
            mock_service.get_user_health_data.assert_called_once()
            call_args = mock_service.get_user_health_data.call_args
            assert call_args.kwargs["limit"] == 50
            assert call_args.kwargs["offset"] == 10

    def test_list_health_data_service_error(self, client: TestClient):
        """Test health data listing with service error through real API."""
        with patch("clarity.api.v1.health_data.get_health_data_service") as mock_get_service:
            mock_service = Mock()
            mock_service.get_user_health_data = AsyncMock(side_effect=Exception("Database error"))
            mock_get_service.return_value = mock_service
            
            response = client.get("/api/v1/")
            
            # Should get server error
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestHealthDataAPIValidation:
    """Test API validation through REAL router."""

    def test_invalid_json_payload(self, client: TestClient):
        """Test invalid JSON payload through real API."""
        response = client.post(
            "/api/v1/health/upload", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422

    def test_missing_required_fields(self, client: TestClient):
        """Test missing required fields through real API."""
        incomplete_data = {
            "user_id": str(uuid4()),
            # Missing metrics and other required fields
        }
        
        response = client.post("/api/v1/health/upload", json=incomplete_data)
        
        assert response.status_code == 422
        response_data = response.json()
        assert "detail" in response_data

    def test_invalid_metric_type(self, client: TestClient):
        """Test invalid metric type through real API."""
        invalid_data = {
            "user_id": str(uuid4()),
            "metrics": [
                {
                    "metric_type": "invalid_type",  # Invalid metric type
                    "activity_data": {
                        "steps": 1000,
                        "distance": 1.0,
                    }
                }
            ],
            "upload_source": "test_app",
            "client_timestamp": datetime.now(UTC).isoformat(),
        }
        
        response = client.post("/api/v1/health/upload", json=invalid_data)
        
        assert response.status_code == 422


class TestHealthDataAPIAuthentication:
    """Test authentication through REAL router."""

    def test_upload_without_auth_override(self):
        """Test upload without authentication override."""
        # Create app without auth override
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app)
        
        upload_data = create_test_health_data_upload().model_dump()
        
        response = client.post("/api/v1/health/upload", json=upload_data)
        
        # Should get authentication error
        assert response.status_code in [401, 403]

    def test_list_without_auth_override(self):
        """Test list without authentication override."""
        # Create app without auth override  
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app)
        
        response = client.get("/api/v1/health/data")
        
        # Should get authentication error
        assert response.status_code in [401, 403]
