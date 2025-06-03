"""Comprehensive tests for Health Data API endpoints.

Tests cover:
- Health data upload endpoints
- Data retrieval with filtering
- Processing status tracking
- Data deletion
- Error handling and validation
- Authentication and authorization
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from fastapi import status
from fastapi.testclient import TestClient
import pytest

from clarity.api.v1.health_data import health_data_router
from clarity.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ValidationError,
)
from clarity.models.health_data import (
    BiometricData,
    HealthDataUpload,
    HealthMetric,
    ProcessingStatus,
)


@pytest.fixture
def mock_health_service():
    """Mock health data service."""
    return AsyncMock()


@pytest.fixture
def mock_auth_service():
    """Mock authentication service."""
    return AsyncMock()


@pytest.fixture
def app_client(mock_health_service, mock_auth_service):
    """Create test client with mocked dependencies."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(health_data_router, prefix="/v1")

    # Mock dependency injection
    with patch("clarity.api.v1.health_data.get_health_data_service", return_value=mock_health_service):
        with patch("clarity.api.v1.health_data.get_auth_service", return_value=mock_auth_service):
            yield TestClient(app), mock_health_service, mock_auth_service


@pytest.fixture
def sample_health_metrics():
    """Sample health metrics for testing."""
    return [
        {
            "metric_id": str(uuid4()),
            "metric_type": "heart_rate",
            "timestamp": datetime.now(UTC).isoformat(),
            "value": 72.0,
            "unit": "bpm",
            "biometric_data": {"heart_rate": 72.0},
        },
        {
            "metric_id": str(uuid4()),
            "metric_type": "step_count",
            "timestamp": datetime.now(UTC).isoformat(),
            "value": 10000.0,
            "unit": "steps",
            "activity_data": {"step_count": 10000},
        },
    ]


@pytest.fixture
def sample_upload_request(sample_health_metrics):
    """Sample health data upload request."""
    return {
        "user_id": str(uuid4()),
        "metrics": sample_health_metrics,
        "upload_source": "apple_health",
        "client_timestamp": datetime.now(UTC).isoformat(),
    }


@pytest.fixture
def auth_headers():
    """Sample authentication headers."""
    return {"Authorization": "Bearer test_token_123"}


class TestHealthDataUploadAPI:
    """Test health data upload endpoints."""

    def test_upload_health_data_success(self, app_client, sample_upload_request, auth_headers):
        """Test successful health data upload."""
        client, mock_service, mock_auth = app_client

        # Mock successful authentication
        mock_auth.validate_token.return_value = {"user_id": sample_upload_request["user_id"]}

        # Mock successful service call
        processing_id = str(uuid4())
        mock_service.process_health_data.return_value = {"processing_id": processing_id}

        response = client.post(
            "/v1/health-data/upload",
            json=sample_upload_request,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_202_ACCEPTED
        response_data = response.json()
        assert response_data["processing_id"] == processing_id
        assert "message" in response_data

    def test_upload_health_data_validation_error(self, app_client, auth_headers):
        """Test health data upload with validation error."""
        client, mock_service, mock_auth = app_client

        # Mock authentication
        mock_auth.validate_token.return_value = {"user_id": str(uuid4())}

        # Invalid request - missing required fields
        invalid_request = {
            "metrics": [],  # Empty metrics should fail validation
            "upload_source": "test",
        }

        response = client.post(
            "/v1/health-data/upload",
            json=invalid_request,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_upload_health_data_service_error(self, app_client, sample_upload_request, auth_headers):
        """Test health data upload with service error."""
        client, mock_service, mock_auth = app_client

        # Mock authentication
        mock_auth.validate_token.return_value = {"user_id": sample_upload_request["user_id"]}

        # Mock service error
        mock_service.process_health_data.side_effect = Exception("Service error")

        response = client.post(
            "/v1/health-data/upload",
            json=sample_upload_request,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_upload_health_data_unauthorized(self, app_client, sample_upload_request):
        """Test health data upload without authentication."""
        client, mock_service, mock_auth = app_client

        response = client.post(
            "/v1/health-data/upload",
            json=sample_upload_request,
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_upload_health_data_invalid_token(self, app_client, sample_upload_request, auth_headers):
        """Test health data upload with invalid token."""
        client, mock_service, mock_auth = app_client

        # Mock authentication failure
        mock_auth.validate_token.side_effect = AuthenticationError("Invalid token")

        response = client.post(
            "/v1/health-data/upload",
            json=sample_upload_request,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_upload_health_data_permission_denied(self, app_client, sample_upload_request, auth_headers):
        """Test health data upload with insufficient permissions."""
        client, mock_service, mock_auth = app_client

        # Mock authentication success but permission denied
        mock_auth.validate_token.return_value = {"user_id": "different_user"}
        mock_auth.check_permission.return_value = False

        response = client.post(
            "/v1/health-data/upload",
            json=sample_upload_request,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestHealthDataRetrievalAPI:
    """Test health data retrieval endpoints."""

    def test_get_user_health_data_success(self, app_client, auth_headers):
        """Test successful user health data retrieval."""
        client, mock_service, mock_auth = app_client

        user_id = str(uuid4())
        mock_auth.validate_token.return_value = {"user_id": user_id}

        # Mock service response
        mock_data = {
            "data": [
                {
                    "metric_type": "heart_rate",
                    "value": 72.0,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ],
            "total_count": 1,
            "has_more": False,
        }
        mock_service.get_user_health_data.return_value = mock_data

        response = client.get(
            f"/v1/health-data/users/{user_id}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["data"] == mock_data["data"]
        assert response_data["total_count"] == 1

    def test_get_user_health_data_with_filters(self, app_client, auth_headers):
        """Test user health data retrieval with filters."""
        client, mock_service, mock_auth = app_client

        user_id = str(uuid4())
        mock_auth.validate_token.return_value = {"user_id": user_id}
        mock_service.get_user_health_data.return_value = {"data": [], "total_count": 0, "has_more": False}

        response = client.get(
            f"/v1/health-data/users/{user_id}",
            params={
                "metric_type": "heart_rate",
                "limit": 50,
                "offset": 10,
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-12-31T23:59:59Z",
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK

        # Verify service was called with correct parameters
        mock_service.get_user_health_data.assert_called_once()
        call_args = mock_service.get_user_health_data.call_args[1]
        assert call_args["metric_type"] == "heart_rate"
        assert call_args["limit"] == 50
        assert call_args["offset"] == 10

    def test_get_user_health_data_unauthorized_access(self, app_client, auth_headers):
        """Test accessing another user's health data."""
        client, mock_service, mock_auth = app_client

        # User tries to access different user's data
        requesting_user_id = str(uuid4())
        target_user_id = str(uuid4())

        mock_auth.validate_token.return_value = {"user_id": requesting_user_id}
        mock_auth.check_permission.return_value = False

        response = client.get(
            f"/v1/health-data/users/{target_user_id}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_get_user_health_data_not_found(self, app_client, auth_headers):
        """Test health data retrieval for non-existent user."""
        client, mock_service, mock_auth = app_client

        user_id = str(uuid4())
        mock_auth.validate_token.return_value = {"user_id": user_id}
        mock_service.get_user_health_data.side_effect = ValidationError("User not found")

        response = client.get(
            f"/v1/health-data/users/{user_id}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_user_health_data_invalid_filters(self, app_client, auth_headers):
        """Test health data retrieval with invalid filters."""
        client, mock_service, mock_auth = app_client

        user_id = str(uuid4())
        mock_auth.validate_token.return_value = {"user_id": user_id}

        response = client.get(
            f"/v1/health-data/users/{user_id}",
            params={
                "limit": -1,  # Invalid limit
                "start_date": "invalid-date",  # Invalid date format
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestProcessingStatusAPI:
    """Test processing status endpoints."""

    def test_get_processing_status_success(self, app_client, auth_headers):
        """Test successful processing status retrieval."""
        client, mock_service, mock_auth = app_client

        user_id = str(uuid4())
        processing_id = str(uuid4())

        mock_auth.validate_token.return_value = {"user_id": user_id}

        # Mock service response
        mock_status = {
            "processing_id": processing_id,
            "status": ProcessingStatus.COMPLETED.value,
            "created_at": datetime.now(UTC).isoformat(),
            "completion_time": datetime.now(UTC).isoformat(),
            "metrics_processed": 100,
        }
        mock_service.get_processing_status.return_value = mock_status

        response = client.get(
            f"/v1/health-data/processing/{processing_id}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["processing_id"] == processing_id
        assert response_data["status"] == ProcessingStatus.COMPLETED.value

    def test_get_processing_status_not_found(self, app_client, auth_headers):
        """Test processing status retrieval for non-existent job."""
        client, mock_service, mock_auth = app_client

        processing_id = str(uuid4())
        mock_auth.validate_token.return_value = {"user_id": str(uuid4())}
        mock_service.get_processing_status.return_value = None

        response = client.get(
            f"/v1/health-data/processing/{processing_id}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_processing_status_permission_denied(self, app_client, auth_headers):
        """Test processing status retrieval with insufficient permissions."""
        client, mock_service, mock_auth = app_client

        processing_id = str(uuid4())
        mock_auth.validate_token.return_value = {"user_id": str(uuid4())}

        # Mock status belonging to different user
        mock_status = {
            "processing_id": processing_id,
            "user_id": str(uuid4()),  # Different user
            "status": "PROCESSING",
        }
        mock_service.get_processing_status.return_value = mock_status

        response = client.get(
            f"/v1/health-data/processing/{processing_id}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestHealthDataDeletionAPI:
    """Test health data deletion endpoints."""

    def test_delete_health_data_success(self, app_client, auth_headers):
        """Test successful health data deletion."""
        client, mock_service, mock_auth = app_client

        user_id = str(uuid4())
        processing_id = str(uuid4())

        mock_auth.validate_token.return_value = {"user_id": user_id}
        mock_service.delete_health_data.return_value = True

        response = client.delete(
            f"/v1/health-data/users/{user_id}",
            params={"processing_id": processing_id},
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["message"] == "Health data deleted successfully"

    def test_delete_all_user_health_data(self, app_client, auth_headers):
        """Test deletion of all user health data."""
        client, mock_service, mock_auth = app_client

        user_id = str(uuid4())
        mock_auth.validate_token.return_value = {"user_id": user_id}
        mock_service.delete_health_data.return_value = True

        response = client.delete(
            f"/v1/health-data/users/{user_id}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        mock_service.delete_health_data.assert_called_with(user_id, None)

    def test_delete_health_data_unauthorized_access(self, app_client, auth_headers):
        """Test deleting another user's health data."""
        client, mock_service, mock_auth = app_client

        requesting_user_id = str(uuid4())
        target_user_id = str(uuid4())

        mock_auth.validate_token.return_value = {"user_id": requesting_user_id}

        response = client.delete(
            f"/v1/health-data/users/{target_user_id}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_delete_health_data_not_found(self, app_client, auth_headers):
        """Test deletion of non-existent health data."""
        client, mock_service, mock_auth = app_client

        user_id = str(uuid4())
        mock_auth.validate_token.return_value = {"user_id": user_id}
        mock_service.delete_health_data.return_value = False

        response = client.delete(
            f"/v1/health-data/users/{user_id}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_health_data_service_error(self, app_client, auth_headers):
        """Test health data deletion with service error."""
        client, mock_service, mock_auth = app_client

        user_id = str(uuid4())
        mock_auth.validate_token.return_value = {"user_id": user_id}
        mock_service.delete_health_data.side_effect = Exception("Deletion failed")

        response = client.delete(
            f"/v1/health-data/users/{user_id}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestHealthDataAPIHealthCheck:
    """Test health check endpoint."""

    def test_health_check_success(self, app_client):
        """Test successful health check."""
        client, mock_service, mock_auth = app_client

        response = client.get("/v1/health-data/health")

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["status"] == "healthy"
        assert "timestamp" in response_data


class TestHealthDataAPIErrorHandling:
    """Test error handling and edge cases."""

    def test_api_handles_concurrent_uploads(self, app_client, sample_upload_request, auth_headers):
        """Test API handles concurrent upload requests."""
        client, mock_service, mock_auth = app_client

        mock_auth.validate_token.return_value = {"user_id": sample_upload_request["user_id"]}
        mock_service.process_health_data.return_value = {"processing_id": str(uuid4())}

        # Start multiple concurrent requests
        responses = []
        for _ in range(3):
            response = client.post(
                "/v1/health-data/upload",
                json=sample_upload_request,
                headers=auth_headers,
            )
            responses.append(response)

        # All should succeed
        assert all(r.status_code == status.HTTP_202_ACCEPTED for r in responses)

    def test_api_validates_request_size(self, app_client, auth_headers):
        """Test API validates request size limits."""
        client, mock_service, mock_auth = app_client

        user_id = str(uuid4())
        mock_auth.validate_token.return_value = {"user_id": user_id}

        # Create oversized request
        large_metrics = []
        for i in range(10000):  # Very large number of metrics
            large_metrics.append({
                "metric_id": str(uuid4()),
                "metric_type": "heart_rate",
                "timestamp": datetime.now(UTC).isoformat(),
                "value": 72.0,
                "unit": "bpm",
            })

        large_request = {
            "user_id": user_id,
            "metrics": large_metrics,
            "upload_source": "test",
            "client_timestamp": datetime.now(UTC).isoformat(),
        }

        response = client.post(
            "/v1/health-data/upload",
            json=large_request,
            headers=auth_headers,
        )

        # Should handle large requests gracefully
        assert response.status_code in [
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ]

    def test_api_handles_malformed_json(self, app_client, auth_headers):
        """Test API handles malformed JSON requests."""
        client, mock_service, mock_auth = app_client

        # Send malformed JSON
        response = client.post(
            "/v1/health-data/upload",
            data="malformed json{",
            headers={**auth_headers, "Content-Type": "application/json"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_api_handles_missing_content_type(self, app_client, sample_upload_request, auth_headers):
        """Test API handles missing content type header."""
        client, mock_service, mock_auth = app_client

        headers_no_content_type = {k: v for k, v in auth_headers.items() if k != "Content-Type"}

        response = client.post(
            "/v1/health-data/upload",
            json=sample_upload_request,
            headers=headers_no_content_type,
        )

        # FastAPI should handle this gracefully
        assert response.status_code in [
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_401_UNAUTHORIZED,  # If auth fails first
        ]

    def test_api_timeout_handling(self, app_client, sample_upload_request, auth_headers):
        """Test API timeout handling for slow services."""
        client, mock_service, mock_auth = app_client

        mock_auth.validate_token.return_value = {"user_id": sample_upload_request["user_id"]}

        # Mock service timeout
        async def slow_service(*args, **kwargs):
            await asyncio.sleep(60)  # Simulate slow service
            return {"processing_id": str(uuid4())}

        mock_service.process_health_data.side_effect = slow_service

        response = client.post(
            "/v1/health-data/upload",
            json=sample_upload_request,
            headers=auth_headers,
        )

        # Should handle timeout gracefully
        assert response.status_code in [
            status.HTTP_504_GATEWAY_TIMEOUT,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ]


class TestHealthDataAPIParameterValidation:
    """Test parameter validation and edge cases."""

    def test_invalid_user_id_format(self, app_client, auth_headers):
        """Test API with invalid user ID format."""
        client, mock_service, mock_auth = app_client

        mock_auth.validate_token.return_value = {"user_id": str(uuid4())}

        response = client.get(
            "/v1/health-data/users/invalid-uuid-format",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_invalid_processing_id_format(self, app_client, auth_headers):
        """Test API with invalid processing ID format."""
        client, mock_service, mock_auth = app_client

        mock_auth.validate_token.return_value = {"user_id": str(uuid4())}

        response = client.get(
            "/v1/health-data/processing/invalid-uuid-format",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_negative_pagination_parameters(self, app_client, auth_headers):
        """Test API with negative pagination parameters."""
        client, mock_service, mock_auth = app_client

        user_id = str(uuid4())
        mock_auth.validate_token.return_value = {"user_id": user_id}

        response = client.get(
            f"/v1/health-data/users/{user_id}",
            params={"limit": -10, "offset": -5},
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_excessive_pagination_limits(self, app_client, auth_headers):
        """Test API with excessive pagination limits."""
        client, mock_service, mock_auth = app_client

        user_id = str(uuid4())
        mock_auth.validate_token.return_value = {"user_id": user_id}

        response = client.get(
            f"/v1/health-data/users/{user_id}",
            params={"limit": 100000},  # Very large limit
            headers=auth_headers,
        )

        # Should either validate limit or handle gracefully
        assert response.status_code in [
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_200_OK,  # If server handles it
        ]
