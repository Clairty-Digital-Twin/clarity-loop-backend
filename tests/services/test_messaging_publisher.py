"""Test messaging publisher module."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from clarity.services.messaging.publisher import (
    HealthDataEvent,
    HealthDataPublisher,
    InsightRequestEvent,
)


class TestHealthDataEvent:
    """Test HealthDataEvent model."""

    def test_health_data_event_creation(self):
        """Test creating health data event."""
        event = HealthDataEvent(
            user_id="user123",
            upload_id="upload456",
            s3_path="s3://bucket/user123/upload456/data.json",
            timestamp="2024-01-01T00:00:00Z",
        )

        assert event.user_id == "user123"
        assert event.upload_id == "upload456"
        assert event.s3_path == "s3://bucket/user123/upload456/data.json"
        assert event.event_type == "health_data_upload"
        assert event.timestamp == "2024-01-01T00:00:00Z"


class TestInsightRequestEvent:
    """Test InsightRequestEvent model."""

    def test_insight_request_event_creation(self):
        """Test creating insight request event."""
        event = InsightRequestEvent(
            user_id="user789",
            upload_id="upload999",
            analysis_results={"heart_rate_avg": 75, "sleep_quality": "good"},
            timestamp="2024-01-02T00:00:00Z",
        )

        assert event.user_id == "user789"
        assert event.upload_id == "upload999"
        assert event.analysis_results == {"heart_rate_avg": 75, "sleep_quality": "good"}
        assert event.event_type == "insight_request"
        assert event.timestamp == "2024-01-02T00:00:00Z"


class TestHealthDataPublisher:
    """Test HealthDataPublisher class."""

    @pytest.mark.asyncio
    async def test_publisher_initialization(self):
        """Test HealthDataPublisher initialization."""
        with patch.dict(
            "os.environ",
            {
                "AWS_REGION": "us-east-1",
                "CLARITY_SNS_TOPIC_ARN": "arn:aws:sns:us-east-1:123456789012:clarity-topic",
                "CLARITY_HEALTH_DATA_QUEUE": "clarity-health-data-processing",
                "CLARITY_INSIGHT_QUEUE": "clarity-insight-generation",
            },
        ):
            with patch(
                "clarity.services.messaging.publisher.AWSMessagingService"
            ) as mock_aws_service:
                publisher = HealthDataPublisher()

                assert publisher.aws_region == "us-east-1"
                assert (
                    publisher.sns_topic_arn
                    == "arn:aws:sns:us-east-1:123456789012:clarity-topic"
                )
                mock_aws_service.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_health_data_upload(self):
        """Test publishing health data upload."""
        with patch.dict(
            "os.environ",
            {
                "AWS_REGION": "us-east-1",
                "CLARITY_SNS_TOPIC_ARN": "arn:aws:sns:us-east-1:123456789012:clarity-topic",
            },
        ):
            with patch(
                "clarity.services.messaging.publisher.AWSMessagingService"
            ) as mock_aws_service:
                mock_messaging_service = Mock()
                mock_messaging_service.publish_health_data_upload = AsyncMock(
                    return_value="msg123"
                )
                mock_aws_service.return_value = mock_messaging_service

                publisher = HealthDataPublisher()

                result = await publisher.publish_health_data_upload(
                    user_id="user123",
                    upload_id="upload456",
                    s3_path="s3://bucket/user123/upload456/data.json",
                    metadata={"source": "apple_health"},
                )

                assert result == "msg123"
                mock_messaging_service.publish_health_data_upload.assert_called_once_with(
                    user_id="user123",
                    upload_id="upload456",
                    s3_path="s3://bucket/user123/upload456/data.json",
                    metadata={"source": "apple_health"},
                )

    @pytest.mark.asyncio
    async def test_publish_insight_request(self):
        """Test publishing insight request."""
        with patch.dict(
            "os.environ",
            {
                "AWS_REGION": "us-east-1",
                "CLARITY_SNS_TOPIC_ARN": "arn:aws:sns:us-east-1:123456789012:clarity-topic",
            },
        ):
            with patch(
                "clarity.services.messaging.publisher.AWSMessagingService"
            ) as mock_aws_service:
                mock_messaging_service = Mock()
                mock_messaging_service.publish_insight_request = AsyncMock(
                    return_value="msg456"
                )
                mock_aws_service.return_value = mock_messaging_service

                publisher = HealthDataPublisher()

                result = await publisher.publish_insight_request(
                    user_id="user789",
                    upload_id="upload999",
                    analysis_results={"heart_rate_avg": 75, "sleep_quality": "good"},
                    metadata={"analysis_version": "1.0"},
                )

                assert result == "msg456"
                mock_messaging_service.publish_insight_request.assert_called_once_with(
                    user_id="user789",
                    upload_id="upload999",
                    analysis_results={"heart_rate_avg": 75, "sleep_quality": "good"},
                    metadata={"analysis_version": "1.0"},
                )
