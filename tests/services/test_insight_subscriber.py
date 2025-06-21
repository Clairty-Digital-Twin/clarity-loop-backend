"""Tests for insight subscriber service."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from clarity.services.messaging.insight_subscriber import InsightSubscriber


class TestInsightSubscriber:
    """Test insight subscriber functionality."""

    @pytest.fixture
    def mock_sqs_client(self):
        """Create mock SQS client."""
        return Mock()

    @pytest.fixture
    def mock_health_repo(self):
        """Create mock health data repository."""
        return AsyncMock()

    @pytest.fixture
    def insight_subscriber(self, mock_sqs_client, mock_health_repo):
        """Create insight subscriber instance."""
        return InsightSubscriber(
            sqs_client=mock_sqs_client,
            queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/test-queue",
            health_data_repository=mock_health_repo,
        )

    def test_initialization(self, insight_subscriber):
        """Test insight subscriber initialization."""
        assert insight_subscriber.queue_url == "https://sqs.us-east-1.amazonaws.com/123456789012/test-queue"
        assert insight_subscriber._running is False

    @pytest.mark.asyncio
    async def test_process_message_invalid_json(self, insight_subscriber, caplog):
        """Test processing invalid JSON message."""
        message = {
            "MessageId": "test-123",
            "Body": "invalid json",
            "ReceiptHandle": "receipt-123",
        }

        await insight_subscriber._process_message(message)

        assert "Failed to parse message body" in caplog.text

    @pytest.mark.asyncio
    async def test_process_message_missing_insight_data(self, insight_subscriber, caplog):
        """Test processing message without insight data."""
        message = {
            "MessageId": "test-123",
            "Body": '{"user_id": "user-123"}',
            "ReceiptHandle": "receipt-123",
        }

        await insight_subscriber._process_message(message)

        assert "Message missing insight_data" in caplog.text

    @pytest.mark.asyncio
    async def test_shutdown(self, insight_subscriber):
        """Test graceful shutdown."""
        insight_subscriber._running = True
        
        await insight_subscriber.shutdown()
        
        assert insight_subscriber._running is False