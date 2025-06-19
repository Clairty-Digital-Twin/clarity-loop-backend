"""Comprehensive tests for insight subscriber service."""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

from clarity.ml.gemini_service import HealthInsightRequest, HealthInsightResponse
from clarity.services.messaging.insight_subscriber import (
    InsightSubscriber,
    InsightSubscriberSingleton,
    get_insight_subscriber,
    insight_app,
)


@pytest.fixture
def mock_storage_client() -> MagicMock:
    """Mock Google Cloud Storage client."""
    with patch("clarity.services.messaging.insight_subscriber.storage.Client") as mock:
        yield mock


@pytest.fixture
def mock_gemini_service() -> MagicMock:
    """Mock Gemini service."""
    with patch("clarity.services.messaging.insight_subscriber.GeminiService") as mock:
        yield mock


@pytest.fixture
def insight_subscriber(
    mock_storage_client: MagicMock,  # noqa: ARG001
    mock_gemini_service: MagicMock,  # noqa: ARG001
) -> InsightSubscriber:
    """Create insight subscriber with mocked dependencies."""
    # Reset singleton
    InsightSubscriberSingleton._instance = None

    return InsightSubscriber()


@pytest.fixture
def test_client() -> TestClient:
    """Create test client for FastAPI app."""
    return TestClient(insight_app)


@pytest.fixture
def valid_pubsub_message():
    """Create a valid Pub/Sub message."""
    message_data = {
        "user_id": "test-user-123",
        "upload_id": "upload-456",
        "analysis_results": {
            "heart_rate": {"average": 72, "risk_level": "normal"},
            "activity": {"steps": 8500, "calories": 350},
        },
        "context": "User is a 35-year-old female",
    }

    encoded_data = base64.b64encode(json.dumps(message_data).encode()).decode()

    return {
        "message": {
            "data": encoded_data,
            "attributes": {"messageType": "insight_request"},
        }
    }


class TestInsightSubscriberInit:
    """Test InsightSubscriber initialization."""

    def test_init_default(
        self, mock_storage_client: MagicMock, mock_gemini_service: MagicMock
    ) -> None:
        """Test initialization with default values."""
        # Clear any existing environment variable
        with patch.dict("os.environ", {}, clear=True):
            subscriber = InsightSubscriber()

            assert subscriber.environment == "development"
            assert subscriber.pubsub_push_audience is None
            mock_storage_client.assert_called_once()
            mock_gemini_service.assert_called_once()

    def test_init_with_environment(
        self, mock_storage_client: MagicMock, mock_gemini_service: MagicMock
    ) -> None:
        """Test initialization with environment variables."""
        with patch.dict(
            "os.environ",
            {
                "ENVIRONMENT": "production",
                "PUBSUB_PUSH_AUDIENCE": "test-audience",
            },
        ):
            subscriber = InsightSubscriber()

            assert subscriber.environment == "production"
            assert subscriber.pubsub_push_audience == "test-audience"


class TestProcessInsightRequest:
    """Test insight request processing."""

    @pytest.mark.asyncio
    async def test_process_insight_request_success(
        self,
        insight_subscriber: InsightSubscriber,
        valid_pubsub_message: dict[str, Any],
    ) -> None:
        """Test successful insight request processing."""
        # Mock request
        mock_request = AsyncMock()
        mock_request.json.return_value = valid_pubsub_message
        mock_request.headers = {}

        # Mock Gemini response with correct fields
        mock_insights = HealthInsightResponse(
            user_id="test-user-123",
            narrative="Your cardiovascular health shows positive trends. Your heart rate is within normal range.",
            key_insights=[
                "Heart rate within normal range",
                "Good cardiovascular health",
            ],
            recommendations=[
                "Continue regular exercise",
                "Maintain current activity levels",
            ],
            confidence_score=0.85,
            generated_at="2024-01-01T12:00:00Z",
        )

        insight_subscriber.gemini_service.generate_health_insights = AsyncMock(
            return_value=mock_insights
        )
        insight_subscriber._store_insights = AsyncMock()

        # Process request
        result = await insight_subscriber.process_insight_request(mock_request)

        # Verify result
        assert result["status"] == "success"
        assert result["user_id"] == "test-user-123"
        assert result["upload_id"] == "upload-456"
        assert result["insights_generated"] is True

        # Verify Gemini was called correctly
        insight_subscriber.gemini_service.generate_health_insights.assert_called_once()
        call_args = (
            insight_subscriber.gemini_service.generate_health_insights.call_args[0][0]
        )
        assert isinstance(call_args, HealthInsightRequest)
        assert call_args.user_id == "test-user-123"
        # Decode the message data to compare with analysis_results
        decoded_data = json.loads(
            base64.b64decode(valid_pubsub_message["message"]["data"])
        )
        assert call_args.analysis_results == decoded_data["analysis_results"]

        # Verify insights were stored
        insight_subscriber._store_insights.assert_called_once_with(
            user_id="test-user-123",
            upload_id="upload-456",
            insights=mock_insights.model_dump(),
        )

    @pytest.mark.asyncio
    async def test_process_insight_request_production_auth(
        self,
        insight_subscriber: InsightSubscriber,
        valid_pubsub_message: dict[str, Any],
    ) -> None:
        """Test insight request processing with production authentication."""
        insight_subscriber.environment = "production"

        # Mock request with auth header
        mock_request = AsyncMock()
        mock_request.json.return_value = valid_pubsub_message
        mock_request.headers = {"authorization": "Bearer test-token"}

        # Mock successful auth verification
        insight_subscriber._verify_pubsub_token = AsyncMock()

        # Mock other dependencies with correct response format
        mock_insights = HealthInsightResponse(
            user_id="test-user-123",
            narrative="Test narrative",
            key_insights=["Test insight"],
            recommendations=["Test recommendation"],
            confidence_score=0.9,
            generated_at="2024-01-01T12:00:00Z",
        )
        insight_subscriber.gemini_service.generate_health_insights = AsyncMock(
            return_value=mock_insights
        )
        insight_subscriber._store_insights = AsyncMock()

        # Process request
        result = await insight_subscriber.process_insight_request(mock_request)

        # Verify auth was checked
        insight_subscriber._verify_pubsub_token.assert_called_once_with(mock_request)
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_process_insight_request_error(
        self, insight_subscriber: InsightSubscriber
    ) -> None:
        """Test insight request processing with error."""
        # Mock request that fails
        mock_request = AsyncMock()
        mock_request.json.side_effect = Exception("JSON parse error")

        # Process request should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await insight_subscriber.process_insight_request(mock_request)

        assert exc_info.value.status_code == 500
        assert "Insight generation failed" in str(exc_info.value.detail)


class TestVerifyPubsubToken:
    """Test Pub/Sub token verification."""

    @pytest.mark.asyncio
    async def test_verify_token_missing_header(
        self, insight_subscriber: InsightSubscriber
    ) -> None:
        """Test token verification with missing header."""
        mock_request = AsyncMock()
        mock_request.headers = {}

        with pytest.raises(HTTPException) as exc_info:
            await insight_subscriber._verify_pubsub_token(mock_request)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Missing authorization header"

    @pytest.mark.asyncio
    async def test_verify_token_valid_bearer(
        self, insight_subscriber: InsightSubscriber
    ) -> None:
        """Test token verification with valid bearer token."""
        mock_request = AsyncMock()
        mock_request.headers = {"authorization": "Bearer valid-token"}

        # Should not raise exception
        await insight_subscriber._verify_pubsub_token(mock_request)

    @pytest.mark.asyncio
    async def test_verify_token_invalid_format(
        self, insight_subscriber: InsightSubscriber
    ) -> None:
        """Test token verification with invalid format."""
        mock_request = AsyncMock()
        mock_request.headers = {"authorization": "Bearer "}

        with pytest.raises(HTTPException) as exc_info:
            await insight_subscriber._verify_pubsub_token(mock_request)

        assert exc_info.value.status_code == 401
        assert "Invalid" in exc_info.value.detail  # Allow either error message format

    @pytest.mark.asyncio
    async def test_verify_token_exception(
        self, insight_subscriber: InsightSubscriber
    ) -> None:
        """Test token verification with unexpected exception."""
        mock_request = AsyncMock()
        # Pass authorization that will trigger the token extraction but be empty
        mock_request.headers = {
            "authorization": " "
        }  # Just a space - will extract empty token

        with pytest.raises(HTTPException) as exc_info:
            await insight_subscriber._verify_pubsub_token(mock_request)

        assert exc_info.value.status_code == 401
        # The empty token triggers _raise_invalid_token_error() which gets caught and re-raised as "Invalid Pub/Sub token"
        assert exc_info.value.detail == "Invalid Pub/Sub token"


class TestExtractMessageData:
    """Test message data extraction."""

    def test_extract_message_data_valid(
        self,
        insight_subscriber: InsightSubscriber,
        valid_pubsub_message: dict[str, Any],
    ) -> None:
        """Test extracting valid message data."""
        message_data = insight_subscriber._extract_message_data(valid_pubsub_message)

        assert message_data["user_id"] == "test-user-123"
        assert message_data["upload_id"] == "upload-456"
        assert "analysis_results" in message_data
        assert message_data["analysis_results"]["heart_rate"]["average"] == 72

    def test_extract_message_data_missing_message(
        self, insight_subscriber: InsightSubscriber
    ) -> None:
        """Test extracting data from invalid message structure."""
        with pytest.raises(HTTPException) as exc_info:
            insight_subscriber._extract_message_data({})

        assert exc_info.value.status_code == 400
        assert "Invalid message format" in str(exc_info.value.detail)

    def test_extract_message_data_invalid_base64(
        self, insight_subscriber: InsightSubscriber
    ) -> None:
        """Test extracting data with invalid base64."""
        pubsub_body = {
            "message": {
                "data": "invalid-base64!!",
            }
        }

        with pytest.raises(HTTPException) as exc_info:
            insight_subscriber._extract_message_data(pubsub_body)

        assert exc_info.value.status_code == 400
        assert "Invalid message format" in str(exc_info.value.detail)

    def test_extract_message_data_invalid_json(
        self, insight_subscriber: InsightSubscriber
    ) -> None:
        """Test extracting data with invalid JSON."""
        pubsub_body = {
            "message": {
                "data": base64.b64encode(b"invalid json").decode(),
            }
        }

        with pytest.raises(HTTPException) as exc_info:
            insight_subscriber._extract_message_data(pubsub_body)

        assert exc_info.value.status_code == 400
        assert "Invalid message format" in str(exc_info.value.detail)

    def test_extract_message_data_missing_field(
        self, insight_subscriber: InsightSubscriber
    ) -> None:
        """Test extracting data with missing required field."""
        message_data = {
            "user_id": "test-user",
            # Missing upload_id and analysis_results
        }

        pubsub_body = {
            "message": {
                "data": base64.b64encode(json.dumps(message_data).encode()).decode(),
            }
        }

        with pytest.raises(HTTPException) as exc_info:
            insight_subscriber._extract_message_data(pubsub_body)

        assert exc_info.value.status_code == 400
        assert "Invalid message format" in str(exc_info.value.detail)


class TestStoreInsights:
    """Test insight storage."""

    @pytest.mark.asyncio
    async def test_store_insights(self, insight_subscriber: InsightSubscriber) -> None:
        """Test storing insights (placeholder implementation)."""
        # Currently just logs, so verify it doesn't raise
        await insight_subscriber._store_insights(
            user_id="test-user",
            upload_id="test-upload",
            insights={"test": "data"},
        )


class TestHelperMethods:
    """Test helper methods."""

    def test_raise_invalid_token_error(self) -> None:
        """Test _raise_invalid_token_error method."""
        with pytest.raises(HTTPException) as exc_info:
            InsightSubscriber._raise_invalid_token_error()

        assert exc_info.value.status_code == 401
        assert "Invalid" in exc_info.value.detail  # Allow either error message format

    def test_raise_missing_field_error(self) -> None:
        """Test _raise_missing_field_error method."""
        with pytest.raises(
            ValueError, match="Missing required field: test_field"
        ) as exc_info:
            InsightSubscriber._raise_missing_field_error("test_field")

        assert str(exc_info.value) == "Missing required field: test_field"


class TestInsightSubscriberSingleton:
    """Test InsightSubscriberSingleton."""

    def test_singleton_get_instance(
        self, mock_storage_client: MagicMock, mock_gemini_service: MagicMock
    ) -> None:
        """Test singleton pattern."""
        # Reset singleton
        InsightSubscriberSingleton._instance = None

        # First call creates instance
        instance1 = InsightSubscriberSingleton.get_instance()
        assert instance1 is not None

        # Second call returns same instance
        instance2 = InsightSubscriberSingleton.get_instance()
        assert instance2 is instance1

    def test_get_insight_subscriber(
        self, mock_storage_client: MagicMock, mock_gemini_service: MagicMock
    ) -> None:
        """Test get_insight_subscriber function."""
        # Reset singleton
        InsightSubscriberSingleton._instance = None

        subscriber = get_insight_subscriber()
        assert isinstance(subscriber, InsightSubscriber)


class TestFastAPIEndpoints:
    """Test FastAPI endpoints."""

    @pytest.mark.asyncio
    async def test_process_insight_task_endpoint(
        self, test_client: TestClient, valid_pubsub_message: dict[str, Any]
    ) -> None:
        """Test /process-task endpoint."""
        with patch(
            "clarity.services.messaging.insight_subscriber.get_insight_subscriber"
        ) as mock_get:
            mock_subscriber = AsyncMock()
            mock_subscriber.process_insight_request.return_value = {
                "status": "success",
                "user_id": "test-user",
            }
            mock_get.return_value = mock_subscriber

            response = test_client.post("/process-task", json=valid_pubsub_message)

            assert response.status_code == 200
            assert response.json()["status"] == "success"

    def test_health_check_endpoint(self, test_client: TestClient) -> None:
        """Test /health endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "service": "insights"}
