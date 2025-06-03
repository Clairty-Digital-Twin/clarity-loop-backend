"""Tests for HealthKit upload API endpoint."""

from datetime import UTC, datetime
import json
from unittest.mock import AsyncMock, Mock, patch
import uuid

from fastapi import HTTPException, status
from fastapi.security import HTTPBearer
from fastapi.testclient import TestClient
import pytest

from clarity.api.v1.healthkit_upload import (
    HealthKitSample,
    HealthKitUploadRequest,
    HealthKitUploadResponse,
    get_auth_scheme,
    get_upload_status,
    router,
    upload_healthkit_data,
)


class TestHealthKitModels:
    """Test the HealthKit data models."""

    def test_healthkit_sample_creation(self) -> None:
        """Test HealthKitSample model creation."""
        sample = HealthKitSample(
            identifier="HKQuantityTypeIdentifierHeartRate",
            type="heart_rate",
            value=75.0,
            unit="count/min",
            start_date="2023-01-01T12:00:00Z",
            end_date="2023-01-01T12:01:00Z",
            source_name="Apple Watch",
        )

        assert sample.identifier == "HKQuantityTypeIdentifierHeartRate"
        assert sample.type == "heart_rate"
        assert sample.value == 75.0
        assert sample.unit == "count/min"
        assert sample.metadata == {}

    def test_healthkit_sample_with_dict_value(self) -> None:
        """Test HealthKitSample with dictionary value."""
        sample = HealthKitSample(
            identifier="HKCategoryTypeIdentifierSleepAnalysis",
            type="sleep",
            value={"stage": "deep", "quality": 0.8},
            start_date="2023-01-01T22:00:00Z",
            end_date="2023-01-01T06:00:00Z",
            metadata={"confidence": 0.95},
        )

        assert isinstance(sample.value, dict)
        assert sample.value["stage"] == "deep"
        assert sample.metadata["confidence"] == 0.95

    def test_upload_request_creation(self) -> None:
        """Test HealthKitUploadRequest model creation."""
        request = HealthKitUploadRequest(
            user_id="test-user-123",
            quantity_samples=[
                HealthKitSample(
                    identifier="HKQuantityTypeIdentifierHeartRate",
                    type="heart_rate",
                    value=75.0,
                    unit="count/min",
                    start_date="2023-01-01T12:00:00Z",
                    end_date="2023-01-01T12:01:00Z",
                )
            ],
            sync_token="token-123",
        )

        assert request.user_id == "test-user-123"
        assert len(request.quantity_samples) == 1
        assert request.sync_token == "token-123"
        assert request.category_samples == []
        assert request.workouts == []

    def test_upload_response_creation(self) -> None:
        """Test HealthKitUploadResponse model creation."""
        response = HealthKitUploadResponse(
            upload_id="test-user-123-abc123",
            status="queued",
            queued_at="2023-01-01T12:00:00Z",
            samples_received={"quantity_samples": 5, "workouts": 2},
            message="Data queued successfully",
        )

        assert response.upload_id == "test-user-123-abc123"
        assert response.status == "queued"
        assert response.samples_received["quantity_samples"] == 5


class TestAuthScheme:
    """Test authentication scheme helper."""

    def test_get_auth_scheme(self) -> None:
        """Test get_auth_scheme returns HTTPBearer instance."""
        auth_scheme = get_auth_scheme()
        assert isinstance(auth_scheme, HTTPBearer)


class TestHealthKitUploadEndpoint:
    """Test the main HealthKit upload endpoint."""

    @pytest.fixture
    def sample_upload_request(self) -> HealthKitUploadRequest:
        """Create a sample upload request for testing."""
        return HealthKitUploadRequest(
            user_id="test-user-123",
            quantity_samples=[
                HealthKitSample(
                    identifier="HKQuantityTypeIdentifierHeartRate",
                    type="heart_rate",
                    value=75.0,
                    unit="count/min",
                    start_date="2023-01-01T12:00:00Z",
                    end_date="2023-01-01T12:01:00Z",
                    source_name="Apple Watch",
                )
            ],
            workouts=[
                {
                    "type": "running",
                    "duration": 1800,
                    "calories": 150,
                    "distance": 2.5,
                }
            ],
            sync_token="sync-token-123",
        )

    @pytest.fixture
    def mock_token(self) -> HTTPBearer:
        """Create a mock authentication token."""
        token = Mock(spec=HTTPBearer)
        token.credentials = "valid-firebase-token"
        return token

    @patch("clarity.auth.firebase_auth.verify_firebase_token")
    @patch("clarity.api.v1.healthkit_upload.storage.Client")
    @patch("clarity.api.v1.healthkit_upload.get_publisher")
    @patch("clarity.api.v1.healthkit_upload.uuid.uuid4")
    @pytest.mark.asyncio
    async def test_successful_upload(
        self,
        mock_uuid: Mock,
        mock_get_publisher: Mock,
        mock_storage_client: Mock,
        mock_verify_token: AsyncMock,
        sample_upload_request: HealthKitUploadRequest,
        mock_token: HTTPBearer,
    ) -> None:
        """Test successful HealthKit data upload."""
        # Setup mocks
        mock_uuid.return_value = Mock(hex="abc123")
        mock_verify_token.return_value = {"uid": "test-user-123"}

        # Mock storage
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        # Mock publisher
        mock_publisher = Mock()
        mock_get_publisher.return_value = mock_publisher

        # Call the endpoint
        result = await upload_healthkit_data(sample_upload_request, mock_token)

        # Verify result
        assert isinstance(result, HealthKitUploadResponse)
        assert result.upload_id == "test-user-123-abc123"
        assert result.status == "queued"
        assert result.samples_received["quantity_samples"] == 1
        assert result.samples_received["workouts"] == 1

        # Verify mocks were called
        mock_verify_token.assert_called_once_with("valid-firebase-token")
        mock_blob.upload_from_string.assert_called_once()
        mock_publisher.publish_health_data_upload.assert_called_once()

    @patch("clarity.auth.firebase_auth.verify_firebase_token")
    @pytest.mark.asyncio
    async def test_upload_forbidden_different_user(
        self,
        mock_verify_token: AsyncMock,
        sample_upload_request: HealthKitUploadRequest,
        mock_token: HTTPBearer,
    ) -> None:
        """Test upload fails when user tries to upload for different user."""
        # Setup mock to return different user ID
        mock_verify_token.return_value = {"uid": "different-user"}

        # Call should raise forbidden error
        with pytest.raises(HTTPException) as exc_info:
            await upload_healthkit_data(sample_upload_request, mock_token)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Cannot upload data for a different user" in str(exc_info.value.detail)

    @patch("clarity.auth.firebase_auth.verify_firebase_token")
    @patch("clarity.api.v1.healthkit_upload.storage.Client")
    @pytest.mark.asyncio
    async def test_upload_storage_error(
        self,
        mock_storage_client: Mock,
        mock_verify_token: AsyncMock,
        sample_upload_request: HealthKitUploadRequest,
        mock_token: HTTPBearer,
    ) -> None:
        """Test upload handles storage errors gracefully."""
        # Setup mocks
        mock_verify_token.return_value = {"uid": "test-user-123"}
        mock_storage_client.side_effect = Exception("Storage unavailable")

        # Call should raise internal server error
        with pytest.raises(HTTPException) as exc_info:
            await upload_healthkit_data(sample_upload_request, mock_token)

        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to process health data upload" in str(exc_info.value.detail)

    @patch("clarity.auth.firebase_auth.verify_firebase_token")
    @pytest.mark.asyncio
    async def test_upload_authentication_error(
        self,
        mock_verify_token: AsyncMock,
        sample_upload_request: HealthKitUploadRequest,
        mock_token: HTTPBearer,
    ) -> None:
        """Test upload handles authentication errors."""
        # Setup mock to raise auth error
        auth_error = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
        mock_verify_token.side_effect = auth_error

        # Call should re-raise the auth error
        with pytest.raises(HTTPException) as exc_info:
            await upload_healthkit_data(sample_upload_request, mock_token)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Invalid token"


class TestUploadStatusEndpoint:
    """Test the upload status endpoint."""

    @pytest.fixture
    def mock_token(self) -> HTTPBearer:
        """Create a mock authentication token."""
        token = Mock(spec=HTTPBearer)
        token.credentials = "valid-firebase-token"
        return token

    @patch("clarity.auth.firebase_auth.verify_firebase_token")
    @pytest.mark.asyncio
    async def test_get_upload_status_success(
        self, mock_verify_token: AsyncMock, mock_token: HTTPBearer
    ) -> None:
        """Test successful upload status retrieval."""
        # Setup mock
        mock_verify_token.return_value = {"uid": "test-user-123"}

        # Call endpoint
        result = await get_upload_status("test-user-123-abc123", mock_token)

        # Verify result
        assert isinstance(result, dict)
        assert result["upload_id"] == "test-user-123-abc123"
        assert result["status"] == "processing"
        assert "progress" in result
        assert "message" in result
        assert "last_updated" in result

    @patch("clarity.auth.firebase_auth.verify_firebase_token")
    @pytest.mark.asyncio
    async def test_get_upload_status_forbidden(
        self, mock_verify_token: AsyncMock, mock_token: HTTPBearer
    ) -> None:
        """Test upload status fails for different user."""
        # Setup mock to return different user
        mock_verify_token.return_value = {"uid": "different-user"}

        # Call should raise forbidden error
        with pytest.raises(HTTPException) as exc_info:
            await get_upload_status("test-user-123-abc123", mock_token)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Access denied to this upload" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_upload_status_invalid_format(
        self, mock_token: HTTPBearer
    ) -> None:
        """Test upload status with invalid upload ID format."""
        # Call with invalid upload ID format
        with pytest.raises(HTTPException) as exc_info:
            await get_upload_status("invalid-format", mock_token)

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid upload ID format" in str(exc_info.value.detail)


class TestHealthKitUploadIntegration:
    """Integration tests for the HealthKit upload router."""

    def test_router_configuration(self) -> None:
        """Test that the router is properly configured."""
        assert router.prefix == "/api/v1/healthkit"
        assert "HealthKit" in router.tags

    def test_router_endpoints(self) -> None:
        """Test that router has expected endpoints."""
        routes = [route.path for route in router.routes]
        assert "/upload" in routes
        assert "/status/{upload_id}" in routes

    @patch("clarity.auth.firebase_auth.verify_firebase_token")
    def test_router_with_test_client(self, mock_verify_token: Mock) -> None:
        """Test router with FastAPI test client."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with TestClient(app) as client:
            # Test that endpoints are accessible
            response = client.get("/api/v1/healthkit/status/test-upload-id")
            # We expect it to fail auth, but the endpoint should be reachable
            assert response.status_code in [401, 422, 500]  # Various possible auth/validation errors
