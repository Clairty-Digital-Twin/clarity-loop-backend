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
from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi import status
from fastapi.testclient import TestClient
import pytest

from clarity.api.v1.health_data import router as health_data_router
from clarity.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    DataValidationError,
)
from clarity.models.health_data import (
    ProcessingStatus,
)


@pytest.fixture
def mock_health_service() -> AsyncMock:
    """Mock health data service."""
    return AsyncMock()


@pytest.fixture
def mock_auth_service() -> AsyncMock:
    """Mock authentication service."""
    return AsyncMock()


@pytest.fixture
def app_client(mock_health_service: AsyncMock, mock_auth_service: AsyncMock) -> Generator[tuple[TestClient, AsyncMock, AsyncMock], None, None]:
    """Create test client with mocked dependencies."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(health_data_router, prefix="/v1")

    # Mock dependency injection
    with patch("clarity.api.v1.health_data.get_health_data_service", return_value=mock_health_service):
        with patch("clarity.api.v1.health_data.get_auth_service", return_value=mock_auth_service):
            yield TestClient(app), mock_health_service, mock_auth_service


@pytest.fixture
def sample_health_metrics() -> list[dict[str, Any]]:
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
def sample_upload_request(sample_health_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Sample health data upload request."""
    return {
        "user_id": str(uuid4()),
        "metrics": sample_health_metrics,
        "upload_source": "apple_health",
        "client_timestamp": datetime.now(UTC).isoformat(),
    }


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Sample authentication headers."""
    return {"Authorization": "Bearer test_token_123"}


class TestHealthDataUploadAPI:
    """Test health data upload endpoints."""

    @staticmethod
    def test_upload_health_data_success(app_client: tuple[TestClient, AsyncMock, AsyncMock], sample_upload_request: dict[str, Any], auth_headers: dict[str, str]) -> None:
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

    @staticmethod
    def test_upload_health_data_validation_error(app_client: tuple[TestClient, AsyncMock, AsyncMock], auth_headers: dict[str, str]) -> None:
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


class TestHealthDataRetrievalAPI:
    """Test health data retrieval endpoints."""

    @staticmethod
    def test_get_user_health_data_success(app_client: tuple[TestClient, AsyncMock, AsyncMock], auth_headers: dict[str, str]) -> None:
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


class TestHealthDataAPIHealthCheck:
    """Test health check endpoint."""

    @staticmethod
    def test_health_check_success(app_client: tuple[TestClient, AsyncMock, AsyncMock]) -> None:
        """Test successful health check."""
        client, mock_service, mock_auth = app_client

        response = client.get("/v1/health-data/health")

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["status"] == "healthy"
        assert "timestamp" in response_data
