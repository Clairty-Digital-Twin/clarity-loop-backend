"""Tests for HealthKit upload API endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
import pytest

from clarity.api.v1.healthkit_upload import (
    HealthKitSample,
    HealthKitUploadRequest,
    HealthKitUploadResponse,
    get_upload_status,
    router,
    upload_healthkit_data,
)
from clarity.auth import UserContext


class TestHealthKitModels:
    """Test the HealthKit data models."""

    @staticmethod
    def test_healthkit_sample_creation() -> None:
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

    @staticmethod
    def test_healthkit_sample_with_dict_value() -> None:
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

    @staticmethod
    def test_upload_request_creation(
        test_env_credentials: dict[str, str],
    ) -> None:  # MODIFIED, noqa: PLR6301
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
            sync_token=test_env_credentials["mock_sync_token"],  # MODIFIED, noqa: S106
        )

        assert request.user_id == "test-user-123"
        assert len(request.quantity_samples) == 1
        assert (
            request.sync_token == test_env_credentials["mock_sync_token"]
        )  # MODIFIED, noqa: S105
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


class TestHealthKitUploadEndpoint:
    """Test the main HealthKit upload endpoint."""

    @pytest.fixture
    @staticmethod
    def sample_upload_request(
        test_env_credentials: dict[str, str],
    ) -> HealthKitUploadRequest:  # MODIFIED, noqa: PLR6301
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
            sync_token=test_env_credentials["mock_sync_token"],  # MODIFIED, noqa: S106
        )

    @pytest.fixture
    def mock_user(self) -> UserContext:
        """Create a mock user context."""
        return UserContext(
            user_id="test-user-123",
            email="test@example.com",
            role="patient",
            permissions=[],
            is_verified=True,
            is_active=True,
        )

    @patch("clarity.api.v1.healthkit_upload.storage.Client")
    @patch("clarity.api.v1.healthkit_upload.get_publisher")
    @patch("clarity.api.v1.healthkit_upload.uuid.uuid4")
    @pytest.mark.asyncio
    async def test_successful_upload(
        self,
        mock_uuid: Mock,
        mock_get_publisher: Mock,
        mock_storage_client: Mock,
        sample_upload_request: HealthKitUploadRequest,
        mock_user: UserContext,
    ) -> None:
        """Test successful HealthKit data upload."""
        # Setup mocks - use proper 32-character hex UUID
        mock_uuid.return_value = Mock(hex="abcdef1234567890abcdef1234567890")

        # Mock storage
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        # Mock publisher
        mock_publisher = AsyncMock()
        mock_get_publisher.return_value = mock_publisher

        # Call the endpoint
        result = await upload_healthkit_data(sample_upload_request, mock_user)

        # Verify result
        assert isinstance(result, HealthKitUploadResponse)
        assert result.upload_id == "test-user-123-abcdef1234567890abcdef1234567890"
        assert result.status == "queued"
        assert result.samples_received["quantity_samples"] == 1
        assert result.samples_received["workouts"] == 1

        # Verify mocks were called
        mock_blob.upload_from_string.assert_called_once()
        mock_publisher.publish_health_data_upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_forbidden_different_user(
        self,
        sample_upload_request: HealthKitUploadRequest,
    ) -> None:
        """Test upload fails when user tries to upload for different user."""
        # Create user with different ID
        different_user = UserContext(
            user_id="different-user",
            email="different@example.com",
            role="patient",
            permissions=[],
            is_verified=True,
            is_active=True,
        )

        # Call should raise forbidden error
        with pytest.raises(HTTPException) as exc_info:
            await upload_healthkit_data(sample_upload_request, different_user)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Cannot upload data for a different user" in str(exc_info.value.detail)

    @patch("clarity.api.v1.healthkit_upload.storage.Client")
    @pytest.mark.asyncio
    async def test_upload_storage_error(
        self,
        mock_storage_client: Mock,
        sample_upload_request: HealthKitUploadRequest,
        mock_user: UserContext,
    ) -> None:
        """Test upload handles storage errors gracefully."""
        # Setup mocks
        mock_storage_client.side_effect = Exception("Storage unavailable")

        # Call should raise internal server error
        with pytest.raises(HTTPException) as exc_info:
            await upload_healthkit_data(sample_upload_request, mock_user)

        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to process health data upload" in str(exc_info.value.detail)

    # Note: Authentication errors are now handled by FastAPI middleware
    # so we don't need to test them at the endpoint level


class TestUploadStatusEndpoint:
    """Test the upload status endpoint."""

    @pytest.fixture
    def mock_user(self) -> UserContext:
        """Create a mock user context."""
        return UserContext(
            user_id="test-user-123",
            email="test@example.com",
            role="patient",
            permissions=[],
            is_verified=True,
            is_active=True,
        )

    @pytest.mark.asyncio
    async def test_get_upload_status_success(self, mock_user: UserContext) -> None:
        """Test successful upload status retrieval."""
        # Call endpoint
        result = await get_upload_status(
            "test-user-123-abcdef1234567890abcdef1234567890", mock_user
        )

        # Verify result
        assert isinstance(result, dict)
        assert result["upload_id"] == "test-user-123-abcdef1234567890abcdef1234567890"
        assert result["status"] == "processing"
        assert "progress" in result
        assert "message" in result
        assert "last_updated" in result

    @pytest.mark.asyncio
    async def test_get_upload_status_forbidden(
        self,
    ) -> None:
        """Test upload status fails for different user."""
        # Create user with different ID
        different_user = UserContext(
            user_id="different-user",
            email="different@example.com",
            role="patient",
            permissions=[],
            is_verified=True,
            is_active=True,
        )

        # Call should raise forbidden error
        with pytest.raises(HTTPException) as exc_info:
            await get_upload_status(
                "test-user-123-abcdef1234567890abcdef1234567890", different_user
            )

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Access denied to this upload" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_upload_status_invalid_format(
        self, mock_user: UserContext
    ) -> None:
        """Test upload status with invalid upload ID format."""
        # Call with invalid upload ID format
        with pytest.raises(HTTPException) as exc_info:
            await get_upload_status("invalid-format", mock_user)

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid upload ID format" in str(exc_info.value.detail)


class TestHealthKitUploadIntegration:
    """Integration tests for the HealthKit upload router."""

    def test_router_configuration(self) -> None:
        """Test that the router is properly configured."""
        # Router prefix is set when included in parent router, not on the router itself
        assert "healthkit" in router.tags

    def test_router_endpoints(self) -> None:
        """Test that router has expected endpoints."""
        # Get path from route objects - handle different route types
        routes = []
        for route in router.routes:
            if hasattr(route, "path"):
                routes.append(route.path)
            elif hasattr(route, "path_format"):
                routes.append(route.path_format)  # type: ignore[attr-defined]

        # Routes include the full prefix path
        assert any("upload" in route_path for route_path in routes)
        assert any("status" in route_path for route_path in routes)

    def test_router_with_test_client(self) -> None:
        """Test router with FastAPI test client."""
        app = FastAPI()
        app.include_router(router, prefix="/healthkit")

        with TestClient(app) as client:
            # Test that endpoints are accessible (will fail auth but endpoint should be reachable)
            response = client.get("/healthkit/status/test-upload-id")
            # We expect it to fail auth/validation, but the endpoint should be reachable
            assert response.status_code in {
                400,  # Bad Request (invalid upload ID format)
                401,  # Unauthorized
                422,  # Validation Error
                500,  # Internal Server Error
            }  # Various possible auth/validation errors
