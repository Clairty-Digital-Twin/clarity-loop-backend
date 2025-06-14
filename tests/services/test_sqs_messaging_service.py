"""Comprehensive tests for SQS messaging service."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError
import pytest

from clarity.services.sqs_messaging_service import (
    HealthDataMessageTypes,
    MessagingError,
    SQSMessagingService,
)


@pytest.fixture
def mock_sqs_client() -> MagicMock:
    """Mock SQS client."""
    return MagicMock()


@pytest.fixture
def mock_sns_client() -> MagicMock:
    """Mock SNS client."""
    return MagicMock()


@pytest.fixture
def sqs_service(
    mock_sqs_client: MagicMock, mock_sns_client: MagicMock
) -> SQSMessagingService:
    """Create SQS service with mocked clients."""
    service = SQSMessagingService(
        queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/test-queue",
        region="us-east-1",
        sns_topic_arn="arn:aws:sns:us-east-1:123456789012:test-topic",
    )
    service.sqs_client = mock_sqs_client
    service.sns_client = mock_sns_client
    return service


@pytest.fixture
def sqs_service_no_sns(mock_sqs_client: MagicMock) -> SQSMessagingService:
    """Create SQS service without SNS."""
    service = SQSMessagingService(
        queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/test-queue",
        region="us-east-1",
    )
    service.sqs_client = mock_sqs_client
    return service


class TestSQSMessagingServiceInit:
    """Test SQS messaging service initialization."""

    def test_init_with_default_params(self) -> None:
        """Test initialization with default parameters."""
        with patch("boto3.client") as mock_boto_client:
            service = SQSMessagingService(
                queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/test-queue"
            )

            assert (
                service.queue_url
                == "https://sqs.us-east-1.amazonaws.com/123456789012/test-queue"
            )
            assert service.region == "us-east-1"
            assert service.sns_topic_arn is None
            mock_boto_client.assert_called_once_with("sqs", region_name="us-east-1")

    def test_init_with_sns(self) -> None:
        """Test initialization with SNS topic."""
        with patch("boto3.client") as mock_boto_client:
            service = SQSMessagingService(
                queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/test-queue",
                sns_topic_arn="arn:aws:sns:us-east-1:123456789012:test-topic",
            )

            assert (
                service.sns_topic_arn == "arn:aws:sns:us-east-1:123456789012:test-topic"
            )
            assert mock_boto_client.call_count == 2
            mock_boto_client.assert_any_call("sqs", region_name="us-east-1")
            mock_boto_client.assert_any_call("sns", region_name="us-east-1")

    def test_init_with_endpoint_url(self) -> None:
        """Test initialization with endpoint URL (LocalStack)."""
        with patch("boto3.client") as mock_boto_client:
            SQSMessagingService(
                queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/test-queue",
                sns_topic_arn="arn:aws:sns:us-east-1:123456789012:test-topic",
                endpoint_url="http://localhost:4566",
            )

            assert mock_boto_client.call_count == 2
            mock_boto_client.assert_any_call(
                "sqs", region_name="us-east-1", endpoint_url="http://localhost:4566"
            )
            mock_boto_client.assert_any_call(
                "sns", region_name="us-east-1", endpoint_url="http://localhost:4566"
            )


class TestPublishMessage:
    """Test message publishing functionality."""

    @pytest.mark.asyncio
    async def test_publish_message_success(
        self, sqs_service: SQSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test successful message publishing."""
        mock_sqs_client.send_message.return_value = {"MessageId": "test-message-id-123"}

        with patch("uuid.uuid4", return_value="test-uuid"):
            message_id = await sqs_service.publish_message(
                message_type="test_type",
                data={"key": "value"},
                attributes={"attr1": "value1"},
            )

        assert message_id == "test-message-id-123"

        # Verify send_message was called correctly
        mock_sqs_client.send_message.assert_called_once()
        call_args = mock_sqs_client.send_message.call_args[1]

        assert call_args["QueueUrl"] == sqs_service.queue_url

        # Parse and verify message body
        message_body = json.loads(call_args["MessageBody"])
        assert message_body["id"] == "test-uuid"
        assert message_body["type"] == "test_type"
        assert message_body["data"] == {"key": "value"}
        assert "timestamp" in message_body

        # Verify message attributes
        assert call_args["MessageAttributes"]["MessageType"] == {
            "DataType": "String",
            "StringValue": "test_type",
        }
        assert call_args["MessageAttributes"]["attr1"] == {
            "DataType": "String",
            "StringValue": "value1",
        }

    @pytest.mark.asyncio
    async def test_publish_message_no_attributes(
        self, sqs_service: SQSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test publishing message without custom attributes."""
        mock_sqs_client.send_message.return_value = {"MessageId": "test-message-id-456"}

        message_id = await sqs_service.publish_message(
            message_type="test_type",
            data={"key": "value"},
        )

        assert message_id == "test-message-id-456"

        # Verify only MessageType attribute is present
        call_args = mock_sqs_client.send_message.call_args[1]
        assert len(call_args["MessageAttributes"]) == 1
        assert "MessageType" in call_args["MessageAttributes"]

    @pytest.mark.asyncio
    async def test_publish_message_client_error(
        self, sqs_service: SQSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test message publishing with ClientError."""
        mock_sqs_client.send_message.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access denied"}},
            "SendMessage",
        )

        with pytest.raises(MessagingError) as exc_info:
            await sqs_service.publish_message(
                message_type="test_type",
                data={"key": "value"},
            )

        assert "Failed to publish message" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_publish_message_unexpected_error(
        self, sqs_service: SQSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test message publishing with unexpected error."""
        mock_sqs_client.send_message.side_effect = Exception("Unexpected error")

        with pytest.raises(MessagingError) as exc_info:
            await sqs_service.publish_message(
                message_type="test_type",
                data={"key": "value"},
            )

        assert "Failed to publish message" in str(exc_info.value)


class TestReceiveMessages:
    """Test message receiving functionality."""

    @pytest.mark.asyncio
    async def test_receive_messages_success(
        self, sqs_service: SQSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test successful message receiving."""
        mock_response = {
            "Messages": [
                {
                    "MessageId": "msg-1",
                    "ReceiptHandle": "receipt-1",
                    "Body": json.dumps(
                        {
                            "id": "1",
                            "type": "test",
                            "data": {"key": "value1"},
                        }
                    ),
                    "MessageAttributes": {"attr1": {"StringValue": "val1"}},
                    "Attributes": {"SentTimestamp": "1234567890"},
                },
                {
                    "MessageId": "msg-2",
                    "ReceiptHandle": "receipt-2",
                    "Body": json.dumps(
                        {
                            "id": "2",
                            "type": "test",
                            "data": {"key": "value2"},
                        }
                    ),
                    "MessageAttributes": {},
                    "Attributes": {},
                },
            ]
        }
        mock_sqs_client.receive_message.return_value = mock_response

        messages = await sqs_service.receive_messages(
            max_messages=5,
            wait_time_seconds=10,
            visibility_timeout=60,
        )

        assert len(messages) == 2
        assert messages[0]["message_id"] == "msg-1"
        assert messages[0]["receipt_handle"] == "receipt-1"
        assert messages[0]["body"]["data"]["key"] == "value1"
        assert messages[0]["attributes"]["attr1"]["StringValue"] == "val1"
        assert messages[0]["system_attributes"]["SentTimestamp"] == "1234567890"

        assert messages[1]["message_id"] == "msg-2"
        assert messages[1]["receipt_handle"] == "receipt-2"
        assert messages[1]["body"]["data"]["key"] == "value2"

        # Verify receive_message was called correctly
        mock_sqs_client.receive_message.assert_called_once_with(
            QueueUrl=sqs_service.queue_url,
            MaxNumberOfMessages=5,
            WaitTimeSeconds=10,
            VisibilityTimeout=60,
            MessageAttributeNames=["All"],
            AttributeNames=["All"],
        )

    @pytest.mark.asyncio
    async def test_receive_messages_empty_queue(
        self, sqs_service: SQSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test receiving from empty queue."""
        mock_sqs_client.receive_message.return_value = {}

        messages = await sqs_service.receive_messages()

        assert messages == []

    @pytest.mark.asyncio
    async def test_receive_messages_json_decode_error(
        self, sqs_service: SQSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test receiving message with invalid JSON."""
        mock_response = {
            "Messages": [
                {
                    "MessageId": "msg-1",
                    "ReceiptHandle": "receipt-1",
                    "Body": "invalid-json",
                },
                {
                    "MessageId": "msg-2",
                    "ReceiptHandle": "receipt-2",
                    "Body": json.dumps({"id": "2", "type": "test", "data": {}}),
                },
            ]
        }
        mock_sqs_client.receive_message.return_value = mock_response

        messages = await sqs_service.receive_messages()

        # Only valid message should be returned
        assert len(messages) == 1
        assert messages[0]["message_id"] == "msg-2"

    @pytest.mark.asyncio
    async def test_receive_messages_client_error(
        self, sqs_service: SQSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test receiving messages with ClientError."""
        mock_sqs_client.receive_message.side_effect = ClientError(
            {"Error": {"Code": "QueueDoesNotExist", "Message": "Queue not found"}},
            "ReceiveMessage",
        )

        with pytest.raises(MessagingError) as exc_info:
            await sqs_service.receive_messages()

        assert "Failed to receive messages" in str(exc_info.value)


class TestDeleteMessage:
    """Test message deletion functionality."""

    @pytest.mark.asyncio
    async def test_delete_message_success(
        self, sqs_service: SQSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test successful message deletion."""
        await sqs_service.delete_message("test-receipt-handle")

        mock_sqs_client.delete_message.assert_called_once_with(
            QueueUrl=sqs_service.queue_url,
            ReceiptHandle="test-receipt-handle",
        )

    @pytest.mark.asyncio
    async def test_delete_message_client_error(
        self, sqs_service: SQSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test message deletion with ClientError."""
        mock_sqs_client.delete_message.side_effect = ClientError(
            {"Error": {"Code": "ReceiptHandleIsInvalid", "Message": "Invalid handle"}},
            "DeleteMessage",
        )

        with pytest.raises(MessagingError) as exc_info:
            await sqs_service.delete_message("invalid-handle")

        assert "Failed to delete message" in str(exc_info.value)


class TestBatchDeleteMessages:
    """Test batch message deletion functionality."""

    @pytest.mark.asyncio
    async def test_batch_delete_messages_success(
        self, sqs_service: SQSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test successful batch deletion."""
        mock_sqs_client.delete_message_batch.return_value = {
            "Successful": [
                {"Id": "0"},
                {"Id": "1"},
            ],
            "Failed": [],
        }

        receipt_handles = ["handle-1", "handle-2"]
        result = await sqs_service.batch_delete_messages(receipt_handles)

        assert len(result["successful"]) == 2
        assert len(result["failed"]) == 0

        # Verify call
        call_args = mock_sqs_client.delete_message_batch.call_args[1]
        assert call_args["QueueUrl"] == sqs_service.queue_url
        assert len(call_args["Entries"]) == 2
        assert call_args["Entries"][0] == {"Id": "0", "ReceiptHandle": "handle-1"}
        assert call_args["Entries"][1] == {"Id": "1", "ReceiptHandle": "handle-2"}

    @pytest.mark.asyncio
    async def test_batch_delete_messages_partial_failure(
        self, sqs_service: SQSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test batch deletion with partial failure."""
        mock_sqs_client.delete_message_batch.return_value = {
            "Successful": [{"Id": "0"}],
            "Failed": [
                {
                    "Id": "1",
                    "Code": "ReceiptHandleIsInvalid",
                    "Message": "Invalid handle",
                }
            ],
        }

        receipt_handles = ["handle-1", "handle-2"]
        result = await sqs_service.batch_delete_messages(receipt_handles)

        assert len(result["successful"]) == 1
        assert len(result["failed"]) == 1

    @pytest.mark.asyncio
    async def test_batch_delete_messages_client_error(
        self, sqs_service: SQSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test batch deletion with ClientError."""
        mock_sqs_client.delete_message_batch.side_effect = ClientError(
            {"Error": {"Code": "BatchRequestTooLong", "Message": "Too many messages"}},
            "DeleteMessageBatch",
        )

        with pytest.raises(MessagingError) as exc_info:
            await sqs_service.batch_delete_messages(["handle-1", "handle-2"])

        assert "Failed to batch delete messages" in str(exc_info.value)


class TestPublishToSNS:
    """Test SNS publishing functionality."""

    @pytest.mark.asyncio
    async def test_publish_to_sns_success(
        self, sqs_service: SQSMessagingService, mock_sns_client: MagicMock
    ) -> None:
        """Test successful SNS publishing."""
        mock_sns_client.publish.return_value = {"MessageId": "sns-message-id-123"}

        message_id = await sqs_service.publish_to_sns(
            subject="Test Subject",
            message={"data": "test"},
            attributes={"attr1": "value1"},
        )

        assert message_id == "sns-message-id-123"

        # Verify publish call
        mock_sns_client.publish.assert_called_once_with(
            TopicArn=sqs_service.sns_topic_arn,
            Subject="Test Subject",
            Message=json.dumps({"data": "test"}),
            MessageAttributes={
                "attr1": {"DataType": "String", "StringValue": "value1"}
            },
        )

    @pytest.mark.asyncio
    async def test_publish_to_sns_no_topic_arn(
        self, sqs_service_no_sns: SQSMessagingService
    ) -> None:
        """Test SNS publishing without topic ARN configured."""
        with pytest.raises(MessagingError) as exc_info:
            await sqs_service_no_sns.publish_to_sns(
                subject="Test",
                message={"data": "test"},
            )

        assert "SNS topic ARN not configured" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_publish_to_sns_client_error(
        self, sqs_service: SQSMessagingService, mock_sns_client: MagicMock
    ) -> None:
        """Test SNS publishing with ClientError."""
        mock_sns_client.publish.side_effect = ClientError(
            {"Error": {"Code": "TopicNotFound", "Message": "Topic not found"}},
            "Publish",
        )

        with pytest.raises(MessagingError) as exc_info:
            await sqs_service.publish_to_sns(
                subject="Test",
                message={"data": "test"},
            )

        assert "Failed to publish to SNS" in str(exc_info.value)


class TestQueueOperations:
    """Test queue-level operations."""

    @pytest.mark.asyncio
    async def test_get_queue_attributes_success(
        self, sqs_service: SQSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test getting queue attributes."""
        mock_sqs_client.get_queue_attributes.return_value = {
            "Attributes": {
                "ApproximateNumberOfMessages": "10",
                "ApproximateNumberOfMessagesNotVisible": "5",
                "CreatedTimestamp": "1234567890",
            }
        }

        attributes = await sqs_service.get_queue_attributes()

        assert attributes["ApproximateNumberOfMessages"] == "10"
        assert attributes["ApproximateNumberOfMessagesNotVisible"] == "5"
        assert attributes["CreatedTimestamp"] == "1234567890"

        mock_sqs_client.get_queue_attributes.assert_called_once_with(
            QueueUrl=sqs_service.queue_url,
            AttributeNames=["All"],
        )

    @pytest.mark.asyncio
    async def test_get_queue_attributes_client_error(
        self, sqs_service: SQSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test getting queue attributes with error."""
        mock_sqs_client.get_queue_attributes.side_effect = ClientError(
            {"Error": {"Code": "QueueDoesNotExist", "Message": "Queue not found"}},
            "GetQueueAttributes",
        )

        with pytest.raises(MessagingError) as exc_info:
            await sqs_service.get_queue_attributes()

        assert "Failed to get queue attributes" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_purge_queue_success(
        self, sqs_service: SQSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test purging queue."""
        await sqs_service.purge_queue()

        mock_sqs_client.purge_queue.assert_called_once_with(
            QueueUrl=sqs_service.queue_url
        )

    @pytest.mark.asyncio
    async def test_purge_queue_client_error(
        self, sqs_service: SQSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test purging queue with error."""
        mock_sqs_client.purge_queue.side_effect = ClientError(
            {"Error": {"Code": "PurgeQueueInProgress", "Message": "Purge in progress"}},
            "PurgeQueue",
        )

        with pytest.raises(MessagingError) as exc_info:
            await sqs_service.purge_queue()

        assert "Failed to purge queue" in str(exc_info.value)


class TestMessagingError:
    """Test MessagingError exception."""

    def test_messaging_error_creation(self) -> None:
        """Test MessagingError initialization."""
        error = MessagingError("Test error", queue_url="test-queue")

        assert str(error) == "[MESSAGING_ERROR] Test error"
        assert error.error_code == "MESSAGING_ERROR"
        assert error.details == {"queue_url": "test-queue"}


class TestHealthDataMessageTypes:
    """Test HealthDataMessageTypes constants."""

    def test_message_types(self) -> None:
        """Test message type constants."""
        assert HealthDataMessageTypes.HEALTH_DATA_UPLOADED == "health_data_uploaded"
        assert HealthDataMessageTypes.ANALYSIS_REQUESTED == "analysis_requested"
        assert HealthDataMessageTypes.ANALYSIS_COMPLETED == "analysis_completed"
        assert HealthDataMessageTypes.INSIGHT_GENERATED == "insight_generated"
        assert HealthDataMessageTypes.ERROR_OCCURRED == "error_occurred"
