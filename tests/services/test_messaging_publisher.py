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
            processing_id="proc456",
            metrics_count=10,
            upload_source="apple_health",
            timestamp="2024-01-01T00:00:00Z",
        )

        assert event.user_id == "user123"
        assert event.processing_id == "proc456"
        assert event.metrics_count == 10
        assert event.upload_source == "apple_health"
        assert event.timestamp == "2024-01-01T00:00:00Z"


class TestInsightRequestEvent:
    """Test InsightRequestEvent model."""

    def test_insight_request_event_creation(self):
        """Test creating insight request event."""
        event = InsightRequestEvent(
            user_id="user789",
            processing_id="proc999",
            metric_types=["heart_rate", "sleep"],
            request_type="summary",
            timestamp="2024-01-02T00:00:00Z",
        )

        assert event.user_id == "user789"
        assert event.processing_id == "proc999"
        assert event.metric_types == ["heart_rate", "sleep"]
        assert event.request_type == "summary"
        assert event.timestamp == "2024-01-02T00:00:00Z"


class TestHealthDataPublisher:
    """Test HealthDataPublisher class."""

    @pytest.mark.asyncio
    async def test_publisher_initialization(self):
        """Test HealthDataPublisher initialization."""
        mock_sns_client = Mock()
        mock_sqs_client = Mock()

        publisher = HealthDataPublisher(
            sns_client=mock_sns_client,
            sqs_client=mock_sqs_client,
            analysis_topic_arn="arn:topic:analysis",
            insight_queue_url="https://sqs.queue.url",
        )

        assert publisher.sns_client == mock_sns_client
        assert publisher.sqs_client == mock_sqs_client
        assert publisher.analysis_topic_arn == "arn:topic:analysis"
        assert publisher.insight_queue_url == "https://sqs.queue.url"

    @pytest.mark.asyncio
    async def test_publish_health_data_for_analysis(self):
        """Test publishing health data for analysis."""
        mock_sns_client = Mock()
        mock_sns_client.publish = AsyncMock(return_value={"MessageId": "123"})

        publisher = HealthDataPublisher(
            sns_client=mock_sns_client,
            sqs_client=Mock(),
            analysis_topic_arn="arn:topic:analysis",
            insight_queue_url="https://sqs.queue.url",
        )

        event = HealthDataEvent(
            user_id="user123",
            processing_id="proc456",
            metrics_count=5,
            upload_source="apple_health",
            timestamp="2024-01-01T00:00:00Z",
        )

        result = await publisher.publish_health_data_for_analysis(event)

        assert result == {"MessageId": "123"}
        mock_sns_client.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_insights(self):
        """Test requesting insights."""
        mock_sqs_client = Mock()
        mock_sqs_client.send_message = AsyncMock(return_value={"MessageId": "456"})

        publisher = HealthDataPublisher(
            sns_client=Mock(),
            sqs_client=mock_sqs_client,
            analysis_topic_arn="arn:topic:analysis",
            insight_queue_url="https://sqs.queue.url",
        )

        event = InsightRequestEvent(
            user_id="user789",
            processing_id="proc999",
            metric_types=["heart_rate"],
            request_type="summary",
            timestamp="2024-01-02T00:00:00Z",
        )

        result = await publisher.request_insights(event)

        assert result == {"MessageId": "456"}
        mock_sqs_client.send_message.assert_called_once()
