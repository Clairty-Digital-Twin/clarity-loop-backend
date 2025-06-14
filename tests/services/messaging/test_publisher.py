"""Comprehensive tests for health data publisher service."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import AsyncMock, Mock, patch
import uuid

import pytest

import clarity.services.messaging.publisher
from clarity.services.messaging.publisher import (
    HealthDataEvent,
    HealthDataPublisher,
    InsightRequestEvent,
    get_publisher,
)


@pytest.fixture
def mock_messaging_service() -> Mock:
    """Mock AWS messaging service."""
    mock = Mock()
    mock.publish_health_data_upload = AsyncMock(return_value="msg-123")
    mock.publish_insight_request = AsyncMock(return_value="msg-456")
    mock.health_check = AsyncMock(return_value={"status": "healthy"})
    mock.close = Mock()
    return mock


@pytest.fixture
def publisher(
    mock_messaging_service: Mock,
) -> Generator[HealthDataPublisher, None, None]:
    """Create publisher with mocked dependencies."""
    with patch("clarity.services.messaging.publisher.AWSMessagingService") as mock_aws:
        mock_aws.return_value = mock_messaging_service
        publisher = HealthDataPublisher()
        yield publisher


class TestHealthDataModels:
    """Test Pydantic models."""

    def test_health_data_event(self) -> None:
        """Test HealthDataEvent model."""
        event = HealthDataEvent(
            user_id="user-123",
            upload_id="upload-456",
            s3_path="s3://bucket/path/data.json",
            timestamp="2024-01-15T12:00:00Z",
            metadata={"source": "mobile"},
        )

        assert event.user_id == "user-123"
        assert event.upload_id == "upload-456"
        assert event.s3_path == "s3://bucket/path/data.json"
        assert event.event_type == "health_data_upload"
        assert event.metadata == {"source": "mobile"}

    def test_insight_request_event(self) -> None:
        """Test InsightRequestEvent model."""
        event = InsightRequestEvent(
            user_id="user-123",
            upload_id="upload-456",
            analysis_results={"risk_score": 0.2},
            timestamp="2024-01-15T12:00:00Z",
            metadata={"model_version": "1.0"},
        )

        assert event.user_id == "user-123"
        assert event.upload_id == "upload-456"
        assert event.analysis_results == {"risk_score": 0.2}
        assert event.event_type == "insight_request"
        assert event.metadata == {"model_version": "1.0"}


class TestHealthDataPublisherInit:
    """Test publisher initialization."""

    def test_init_default_env(self) -> None:
        """Test initialization with default environment."""
        with patch(
            "clarity.services.messaging.publisher.AWSMessagingService"
        ) as mock_aws:
            _ = HealthDataPublisher()  # Publisher is created to test initialization

            mock_aws.assert_called_once_with(
                region="us-east-1",
                health_data_queue="clarity-health-data-processing",
                insight_queue="clarity-insight-generation",
                sns_topic_arn=None,
            )

    @patch.dict(
        "os.environ",
        {
            "AWS_REGION": "eu-west-1",
            "CLARITY_SNS_TOPIC_ARN": "arn:aws:sns:eu-west-1:123456789012:clarity-topic",
            "CLARITY_HEALTH_DATA_QUEUE": "custom-health-queue",
            "CLARITY_INSIGHT_QUEUE": "custom-insight-queue",
        },
    )
    def test_init_custom_env(self) -> None:
        """Test initialization with custom environment variables."""
        with patch(
            "clarity.services.messaging.publisher.AWSMessagingService"
        ) as mock_aws:
            publisher = HealthDataPublisher()

            assert publisher.aws_region == "eu-west-1"
            assert (
                publisher.sns_topic_arn
                == "arn:aws:sns:eu-west-1:123456789012:clarity-topic"
            )

            mock_aws.assert_called_once_with(
                region="eu-west-1",
                health_data_queue="custom-health-queue",
                insight_queue="custom-insight-queue",
                sns_topic_arn="arn:aws:sns:eu-west-1:123456789012:clarity-topic",
            )


class TestPublishHealthDataUpload:
    """Test health data upload publishing."""

    @pytest.mark.asyncio
    async def test_publish_health_data_upload_success(
        self, publisher: HealthDataPublisher, mock_messaging_service: Mock
    ) -> None:
        """Test successful health data upload publishing."""
        user_id = str(uuid.uuid4())
        upload_id = str(uuid.uuid4())
        s3_path = "s3://health-bucket/raw/data.json"
        metadata = {"source": "mobile_app", "version": "2.0"}

        message_id = await publisher.publish_health_data_upload(
            user_id=user_id,
            upload_id=upload_id,
            s3_path=s3_path,
            metadata=metadata,
        )

        assert message_id == "msg-123"
        mock_messaging_service.publish_health_data_upload.assert_called_once_with(
            user_id=user_id,
            upload_id=upload_id,
            s3_path=s3_path,
            metadata=metadata,
        )

    @pytest.mark.asyncio
    async def test_publish_health_data_upload_no_metadata(
        self, publisher: HealthDataPublisher, mock_messaging_service: Mock
    ) -> None:
        """Test publishing without metadata."""
        message_id = await publisher.publish_health_data_upload(
            user_id="user-123",
            upload_id="upload-456",
            s3_path="s3://bucket/data.json",
        )

        assert message_id == "msg-123"
        mock_messaging_service.publish_health_data_upload.assert_called_once_with(
            user_id="user-123",
            upload_id="upload-456",
            s3_path="s3://bucket/data.json",
            metadata=None,
        )

    @pytest.mark.asyncio
    async def test_publish_health_data_upload_error(
        self, publisher: HealthDataPublisher, mock_messaging_service: Mock
    ) -> None:
        """Test publishing with error."""
        mock_messaging_service.publish_health_data_upload.side_effect = Exception(
            "SQS error"
        )

        with pytest.raises(Exception, match="SQS error"):
            await publisher.publish_health_data_upload(
                user_id="user-123",
                upload_id="upload-456",
                s3_path="s3://bucket/data.json",
            )


class TestPublishInsightRequest:
    """Test insight request publishing."""

    @pytest.mark.asyncio
    async def test_publish_insight_request_success(
        self, publisher: HealthDataPublisher, mock_messaging_service: Mock
    ) -> None:
        """Test successful insight request publishing."""
        user_id = str(uuid.uuid4())
        upload_id = str(uuid.uuid4())
        analysis_results = {
            "risk_score": 0.2,
            "health_status": "good",
            "metrics_analyzed": 15,
        }
        metadata = {"model_version": "1.0", "processing_time": 2.5}

        message_id = await publisher.publish_insight_request(
            user_id=user_id,
            upload_id=upload_id,
            analysis_results=analysis_results,
            metadata=metadata,
        )

        assert message_id == "msg-456"
        mock_messaging_service.publish_insight_request.assert_called_once_with(
            user_id=user_id,
            upload_id=upload_id,
            analysis_results=analysis_results,
            metadata=metadata,
        )

    @pytest.mark.asyncio
    async def test_publish_insight_request_no_metadata(
        self, publisher: HealthDataPublisher, mock_messaging_service: Mock
    ) -> None:
        """Test publishing without metadata."""
        message_id = await publisher.publish_insight_request(
            user_id="user-123",
            upload_id="upload-456",
            analysis_results={"status": "complete"},
        )

        assert message_id == "msg-456"
        mock_messaging_service.publish_insight_request.assert_called_once_with(
            user_id="user-123",
            upload_id="upload-456",
            analysis_results={"status": "complete"},
            metadata=None,
        )

    @pytest.mark.asyncio
    async def test_publish_insight_request_error(
        self, publisher: HealthDataPublisher, mock_messaging_service: Mock
    ) -> None:
        """Test publishing with error."""
        mock_messaging_service.publish_insight_request.side_effect = Exception(
            "SQS error"
        )

        with pytest.raises(Exception, match="SQS error"):
            await publisher.publish_insight_request(
                user_id="user-123",
                upload_id="upload-456",
                analysis_results={"test": "data"},
            )


class TestHealthCheckAndClose:
    """Test health check and close methods."""

    @pytest.mark.asyncio
    async def test_health_check(
        self, publisher: HealthDataPublisher, mock_messaging_service: Mock
    ) -> None:
        """Test health check."""
        result = await publisher.health_check()

        assert result == {"status": "healthy"}
        mock_messaging_service.health_check.assert_called_once()

    def test_close(
        self, publisher: HealthDataPublisher, mock_messaging_service: Mock
    ) -> None:
        """Test close method."""
        publisher.close()

        mock_messaging_service.close.assert_called_once()


class TestGetPublisher:
    """Test get_publisher singleton function."""

    @pytest.mark.asyncio
    async def test_get_publisher_singleton(self) -> None:
        """Test get_publisher returns singleton."""
        # Reset global state
        clarity.services.messaging.publisher._publisher = None

        with patch("clarity.services.messaging.publisher.AWSMessagingService"):
            publisher1 = await get_publisher()
            publisher2 = await get_publisher()

            assert publisher1 is publisher2

    @pytest.mark.asyncio
    async def test_get_publisher_creates_instance(self) -> None:
        """Test get_publisher creates new instance when needed."""
        # Reset global state
        clarity.services.messaging.publisher._publisher = None

        with patch(
            "clarity.services.messaging.publisher.AWSMessagingService"
        ) as mock_aws:
            publisher = await get_publisher()

            assert publisher is not None
            mock_aws.assert_called_once()


class TestLoggingIntegration:
    """Test logging behavior."""

    @pytest.mark.asyncio
    async def test_publish_health_data_logs_success(
        self, publisher: HealthDataPublisher, mock_messaging_service: Mock
    ) -> None:
        """Test successful publish logs info."""
        # Patch the logger in the instance instead of the module
        with patch.object(publisher, "logger") as mock_logger:
            await publisher.publish_health_data_upload(
                user_id="user-123",
                upload_id="upload-456",
                s3_path="s3://bucket/data.json",
            )

            # Should log success
            mock_logger.info.assert_called()
            call_args = mock_logger.info.call_args[0]
            assert "Published health data upload event" in call_args[0]
            assert "user-123" in str(call_args)
            assert "upload-456" in str(call_args)
            assert "msg-123" in str(call_args)

    @pytest.mark.asyncio
    async def test_publish_health_data_logs_error(
        self, publisher: HealthDataPublisher, mock_messaging_service: Mock
    ) -> None:
        """Test failed publish logs exception."""
        mock_messaging_service.publish_health_data_upload.side_effect = Exception(
            "Test error"
        )

        # Patch the logger in the instance instead of the module
        with patch.object(publisher, "logger") as mock_logger:
            with pytest.raises(Exception, match="Test error"):
                await publisher.publish_health_data_upload(
                    user_id="user-123",
                    upload_id="upload-456",
                    s3_path="s3://bucket/data.json",
                )

            # Should log exception
            mock_logger.exception.assert_called_once()
            assert (
                "Failed to publish health data event"
                in mock_logger.exception.call_args[0][0]
            )
