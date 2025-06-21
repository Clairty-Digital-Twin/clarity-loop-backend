"""Tests for SQS messaging service - real functionality tests."""

from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from clarity.services.sqs_messaging_service import MessagingError, SQSMessagingService


class TestSQSMessagingService:
    """Test SQS messaging service real functionality."""

    @pytest.fixture
    def mock_sqs_client(self):
        """Create a mock SQS client with realistic responses."""
        client = Mock()
        # Default successful response for send_message
        client.send_message.return_value = {
            "MessageId": "test-message-123",
            "MD5OfMessageBody": "abc123",
        }
        # Default successful response for receive_message
        client.receive_message.return_value = {
            "Messages": [
                {
                    "MessageId": "msg-123",
                    "ReceiptHandle": "receipt-123",
                    "Body": '{"test": "data"}',
                }
            ]
        }
        return client

    @pytest.fixture
    def sqs_service(self, mock_sqs_client):
        """Create SQS service with mocked client."""
        service = SQSMessagingService(
            queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/test-queue",
            region="us-east-1",
        )
        # Replace the client with our mock
        service.sqs_client = mock_sqs_client
        return service

    def test_initialization(self):
        """Test SQS service initialization with real parameters."""
        service = SQSMessagingService(
            queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/my-queue",
            region="us-west-2",
            sns_topic_arn="arn:aws:sns:us-west-2:123456789012:my-topic",
        )
        
        assert service.queue_url == "https://sqs.us-east-1.amazonaws.com/123456789012/my-queue"
        assert service.region == "us-west-2"
        assert service.sns_topic_arn == "arn:aws:sns:us-west-2:123456789012:my-topic"

    @pytest.mark.asyncio
    async def test_publish_message_success(self, sqs_service, mock_sqs_client):
        """Test successful message publishing."""
        message_data = {"user_id": "user-123", "action": "analyze_data"}
        
        message_id = await sqs_service.publish_message(message_data)
        
        # Verify the message was sent correctly
        assert message_id == "test-message-123"
        mock_sqs_client.send_message.assert_called_once()
        
        # Check the call arguments
        call_args = mock_sqs_client.send_message.call_args[1]
        assert call_args["QueueUrl"] == sqs_service.queue_url
        assert "MessageBody" in call_args
        
        # Verify message body is valid JSON
        import json
        body = json.loads(call_args["MessageBody"])
        assert body["user_id"] == "user-123"
        assert body["action"] == "analyze_data"

    @pytest.mark.asyncio
    async def test_publish_message_with_attributes(self, sqs_service, mock_sqs_client):
        """Test publishing message with attributes."""
        message_data = {"data": "test"}
        attributes = {
            "Priority": {"StringValue": "High", "DataType": "String"},
            "Timestamp": {"StringValue": "2024-01-01", "DataType": "String"},
        }
        
        await sqs_service.publish_message(message_data, message_attributes=attributes)
        
        call_args = mock_sqs_client.send_message.call_args[1]
        assert call_args["MessageAttributes"] == attributes

    @pytest.mark.asyncio
    async def test_publish_message_client_error(self, sqs_service, mock_sqs_client):
        """Test handling of AWS client errors."""
        mock_sqs_client.send_message.side_effect = ClientError(
            {"Error": {"Code": "InvalidParameterValue", "Message": "Invalid queue URL"}},
            "SendMessage"
        )
        
        with pytest.raises(MessagingError) as exc_info:
            await sqs_service.publish_message({"test": "data"})
        
        assert "Failed to publish message" in str(exc_info.value)
        assert exc_info.value.error_code == "MESSAGING_ERROR"

    @pytest.mark.asyncio
    async def test_receive_messages_success(self, sqs_service, mock_sqs_client):
        """Test successful message reception."""
        messages = await sqs_service.receive_messages(max_messages=5)
        
        assert len(messages) == 1
        assert messages[0]["MessageId"] == "msg-123"
        assert messages[0]["Body"] == '{"test": "data"}'
        
        mock_sqs_client.receive_message.assert_called_once_with(
            QueueUrl=sqs_service.queue_url,
            MaxNumberOfMessages=5,
            WaitTimeSeconds=20,  # Long polling
        )

    @pytest.mark.asyncio
    async def test_receive_messages_empty_queue(self, sqs_service, mock_sqs_client):
        """Test receiving from empty queue."""
        mock_sqs_client.receive_message.return_value = {}  # No Messages key
        
        messages = await sqs_service.receive_messages()
        
        assert messages == []

    @pytest.mark.asyncio
    async def test_delete_message_success(self, sqs_service, mock_sqs_client):
        """Test successful message deletion."""
        receipt_handle = "receipt-handle-123"
        
        await sqs_service.delete_message(receipt_handle)
        
        mock_sqs_client.delete_message.assert_called_once_with(
            QueueUrl=sqs_service.queue_url,
            ReceiptHandle=receipt_handle,
        )

    @pytest.mark.asyncio
    async def test_delete_message_error(self, sqs_service, mock_sqs_client):
        """Test error handling in message deletion."""
        mock_sqs_client.delete_message.side_effect = ClientError(
            {"Error": {"Code": "ReceiptHandleIsInvalid", "Message": "Invalid receipt"}},
            "DeleteMessage"
        )
        
        with pytest.raises(MessagingError) as exc_info:
            await sqs_service.delete_message("bad-receipt")
        
        assert "Failed to delete message" in str(exc_info.value)