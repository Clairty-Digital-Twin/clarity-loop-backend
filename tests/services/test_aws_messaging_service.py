"""Comprehensive tests for AWS messaging service."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError
import pytest

from clarity.services.aws_messaging_service import (
    AWSMessagingService,
    HealthDataEvent,
    InsightRequestEvent,
    get_messaging_service,
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
def aws_messaging_service(
    mock_sqs_client: MagicMock, mock_sns_client: MagicMock
) -> AWSMessagingService:
    """Create AWS messaging service with mocked clients."""
    service = AWSMessagingService(
        region="us-east-1",
        health_data_queue="test-health-queue",
        insight_queue="test-insight-queue",
        sns_topic_arn="arn:aws:sns:us-east-1:123456789012:test-topic",
    )
    service.sqs_client = mock_sqs_client
    service.sns_client = mock_sns_client
    service._queue_urls = {
        "test-health-queue": "https://sqs.us-east-1.amazonaws.com/123456789012/test-health-queue",
        "test-insight-queue": "https://sqs.us-east-1.amazonaws.com/123456789012/test-insight-queue",
    }
    return service


@pytest.fixture
def aws_messaging_service_no_sns(mock_sqs_client: MagicMock) -> AWSMessagingService:
    """Create AWS messaging service without SNS."""
    service = AWSMessagingService(
        region="us-east-1",
        health_data_queue="test-health-queue",
        insight_queue="test-insight-queue",
    )
    service.sqs_client = mock_sqs_client
    service._queue_urls = {}
    return service


class TestAWSMessagingServiceInit:
    """Test AWS messaging service initialization."""

    def test_init_with_default_params(self) -> None:
        """Test initialization with default parameters."""
        with patch("boto3.client") as mock_boto_client:
            service = AWSMessagingService()

            assert service.region == "us-east-1"
            assert service.endpoint_url is None
            assert service.health_data_queue == "clarity-health-data-processing"
            assert service.insight_queue == "clarity-insight-generation"
            assert service.sns_topic_arn is None

            mock_boto_client.assert_called_once_with(
                "sqs",
                region_name="us-east-1",
                endpoint_url=None,
            )

    def test_init_with_sns(self) -> None:
        """Test initialization with SNS topic."""
        with patch("boto3.client") as mock_boto_client:
            service = AWSMessagingService(
                sns_topic_arn="arn:aws:sns:us-east-1:123456789012:test-topic"
            )

            assert (
                service.sns_topic_arn == "arn:aws:sns:us-east-1:123456789012:test-topic"
            )
            assert mock_boto_client.call_count == 2
            mock_boto_client.assert_any_call(
                "sqs", region_name="us-east-1", endpoint_url=None
            )
            mock_boto_client.assert_any_call(
                "sns", region_name="us-east-1", endpoint_url=None
            )

    def test_init_with_endpoint_url(self) -> None:
        """Test initialization with endpoint URL (LocalStack)."""
        with patch("boto3.client") as mock_boto_client:
            service = AWSMessagingService(
                endpoint_url="http://localhost:4566",
                sns_topic_arn="arn:aws:sns:us-east-1:123456789012:test-topic",
            )

            assert service.endpoint_url == "http://localhost:4566"
            mock_boto_client.assert_any_call(
                "sqs", region_name="us-east-1", endpoint_url="http://localhost:4566"
            )
            mock_boto_client.assert_any_call(
                "sns", region_name="us-east-1", endpoint_url="http://localhost:4566"
            )


class TestGetQueueUrl:
    """Test queue URL retrieval functionality."""

    @pytest.mark.asyncio
    async def test_get_queue_url_cached(
        self, aws_messaging_service: AWSMessagingService
    ) -> None:
        """Test getting queue URL from cache."""
        url = await aws_messaging_service._get_queue_url("test-health-queue")

        assert (
            url == "https://sqs.us-east-1.amazonaws.com/123456789012/test-health-queue"
        )
        # Should not call SQS since it's cached
        aws_messaging_service.sqs_client.get_queue_url.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_queue_url_existing_queue(
        self,
        aws_messaging_service_no_sns: AWSMessagingService,
        mock_sqs_client: MagicMock,
    ) -> None:
        """Test getting URL for existing queue."""
        mock_sqs_client.get_queue_url.return_value = {
            "QueueUrl": "https://sqs.us-east-1.amazonaws.com/123456789012/existing-queue"
        }

        url = await aws_messaging_service_no_sns._get_queue_url("existing-queue")

        assert url == "https://sqs.us-east-1.amazonaws.com/123456789012/existing-queue"
        mock_sqs_client.get_queue_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_queue_url_create_new_queue(
        self,
        aws_messaging_service_no_sns: AWSMessagingService,
        mock_sqs_client: MagicMock,
    ) -> None:
        """Test creating new queue when it doesn't exist."""
        mock_sqs_client.get_queue_url.side_effect = ClientError(
            {"Error": {"Code": "AWS.SimpleQueueService.NonExistentQueue"}},
            "GetQueueUrl",
        )
        mock_sqs_client.create_queue.return_value = {
            "QueueUrl": "https://sqs.us-east-1.amazonaws.com/123456789012/new-queue"
        }

        url = await aws_messaging_service_no_sns._get_queue_url("new-queue")

        assert url == "https://sqs.us-east-1.amazonaws.com/123456789012/new-queue"
        mock_sqs_client.create_queue.assert_called_once_with(
            QueueName="new-queue",
            Attributes={
                "MessageRetentionPeriod": "1209600",
                "VisibilityTimeout": "300",
                "ReceiveMessageWaitTimeSeconds": "20",
            },
        )

    @pytest.mark.asyncio
    async def test_get_queue_url_other_error(
        self,
        aws_messaging_service_no_sns: AWSMessagingService,
        mock_sqs_client: MagicMock,
    ) -> None:
        """Test handling other client errors."""
        mock_sqs_client.get_queue_url.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "GetQueueUrl"
        )

        with pytest.raises(ClientError):
            await aws_messaging_service_no_sns._get_queue_url("test-queue")


class TestPublishHealthDataUpload:
    """Test health data upload publishing."""

    @pytest.mark.asyncio
    async def test_publish_health_data_upload_success(
        self, aws_messaging_service: AWSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test successful health data upload publishing."""
        mock_sqs_client.send_message.return_value = {"MessageId": "msg-123"}

        user_id = "user-123"
        upload_id = "upload-456"
        s3_path = "s3://bucket/path/data.json"
        metadata = {"source": "mobile_app"}

        message_id = await aws_messaging_service.publish_health_data_upload(
            user_id, upload_id, s3_path, metadata
        )

        assert message_id == "msg-123"

        # Verify SQS message
        mock_sqs_client.send_message.assert_called_once()
        call_args = mock_sqs_client.send_message.call_args[1]

        assert (
            call_args["QueueUrl"]
            == "https://sqs.us-east-1.amazonaws.com/123456789012/test-health-queue"
        )

        body = json.loads(call_args["MessageBody"])
        assert body["user_id"] == user_id
        assert body["upload_id"] == upload_id
        assert body["s3_path"] == s3_path
        assert body["metadata"] == metadata
        assert "timestamp" in body

        # Check message attributes
        attrs = call_args["MessageAttributes"]
        assert attrs["user_id"]["StringValue"] == user_id
        assert attrs["upload_id"]["StringValue"] == upload_id
        assert attrs["event_type"]["StringValue"] == "health_data_upload"

    @pytest.mark.asyncio
    async def test_publish_health_data_upload_with_sns(
        self,
        aws_messaging_service: AWSMessagingService,
        mock_sqs_client: MagicMock,
        mock_sns_client: MagicMock,
    ) -> None:
        """Test publishing with SNS fan-out."""
        mock_sqs_client.send_message.return_value = {"MessageId": "msg-123"}
        mock_sns_client.publish.return_value = {"MessageId": "sns-msg-456"}

        message_id = await aws_messaging_service.publish_health_data_upload(
            "user-123", "upload-456", "s3://bucket/data.json"
        )

        assert message_id == "msg-123"
        mock_sqs_client.send_message.assert_called_once()
        mock_sns_client.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_health_data_upload_error(
        self, aws_messaging_service: AWSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test handling publish errors."""
        mock_sqs_client.send_message.side_effect = Exception("SQS error")

        with pytest.raises(Exception, match="SQS error"):
            await aws_messaging_service.publish_health_data_upload(
                "user-123", "upload-456", "s3://bucket/data.json"
            )


class TestPublishInsightRequest:
    """Test insight request publishing."""

    @pytest.mark.asyncio
    async def test_publish_insight_request_success(
        self, aws_messaging_service: AWSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test successful insight request publishing."""
        mock_sqs_client.send_message.return_value = {"MessageId": "msg-789"}

        user_id = "user-123"
        upload_id = "upload-456"
        analysis_results = {"risk_score": 0.2, "health_status": "good"}
        metadata = {"model_version": "1.0"}

        message_id = await aws_messaging_service.publish_insight_request(
            user_id, upload_id, analysis_results, metadata
        )

        assert message_id == "msg-789"

        # Verify SQS message
        mock_sqs_client.send_message.assert_called_once()
        call_args = mock_sqs_client.send_message.call_args[1]

        assert (
            call_args["QueueUrl"]
            == "https://sqs.us-east-1.amazonaws.com/123456789012/test-insight-queue"
        )

        body = json.loads(call_args["MessageBody"])
        assert body["user_id"] == user_id
        assert body["upload_id"] == upload_id
        assert body["analysis_results"] == analysis_results
        assert body["metadata"] == metadata

    @pytest.mark.asyncio
    async def test_publish_insight_request_with_sns(
        self,
        aws_messaging_service: AWSMessagingService,
        mock_sqs_client: MagicMock,
        mock_sns_client: MagicMock,
    ) -> None:
        """Test publishing with SNS fan-out."""
        mock_sqs_client.send_message.return_value = {"MessageId": "msg-789"}
        mock_sns_client.publish.return_value = {"MessageId": "sns-msg-999"}

        message_id = await aws_messaging_service.publish_insight_request(
            "user-123", "upload-456", {"test": "results"}
        )

        assert message_id == "msg-789"
        mock_sns_client.publish.assert_called_once()


class TestPublishToSNS:
    """Test SNS publishing functionality."""

    @pytest.mark.asyncio
    async def test_publish_to_sns_success(
        self, aws_messaging_service: AWSMessagingService, mock_sns_client: MagicMock
    ) -> None:
        """Test successful SNS publishing."""
        mock_sns_client.publish.return_value = {"MessageId": "sns-123"}

        message_id = await aws_messaging_service._publish_to_sns(
            subject="Test Subject",
            message={"data": "test"},
            attributes={"key": "value"},
        )

        assert message_id == "sns-123"

        mock_sns_client.publish.assert_called_once_with(
            TopicArn="arn:aws:sns:us-east-1:123456789012:test-topic",
            Subject="Test Subject",
            Message=json.dumps({"data": "test"}, indent=2),
            MessageAttributes={"key": {"DataType": "String", "StringValue": "value"}},
        )

    @pytest.mark.asyncio
    async def test_publish_to_sns_no_sns_configured(
        self, aws_messaging_service_no_sns: AWSMessagingService
    ) -> None:
        """Test SNS publishing when not configured."""
        with pytest.raises(RuntimeError, match="SNS not configured"):
            await aws_messaging_service_no_sns._publish_to_sns(
                subject="Test", message={"data": "test"}
            )

    @pytest.mark.asyncio
    async def test_publish_to_sns_error(
        self, aws_messaging_service: AWSMessagingService, mock_sns_client: MagicMock
    ) -> None:
        """Test SNS publishing with error."""
        mock_sns_client.publish.side_effect = Exception("SNS error")

        with pytest.raises(Exception, match="SNS error"):
            await aws_messaging_service._publish_to_sns(
                subject="Test", message={"data": "test"}
            )


class TestReceiveMessages:
    """Test message receiving functionality."""

    @pytest.mark.asyncio
    async def test_receive_messages_success(
        self, aws_messaging_service: AWSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test successful message receiving."""
        mock_sqs_client.receive_message.return_value = {
            "Messages": [
                {
                    "MessageId": "msg-1",
                    "Body": json.dumps({"data": "test1"}),
                    "ReceiptHandle": "receipt-1",
                },
                {
                    "MessageId": "msg-2",
                    "Body": json.dumps({"data": "test2"}),
                    "ReceiptHandle": "receipt-2",
                },
            ]
        }

        messages = await aws_messaging_service.receive_messages(
            "test-health-queue", max_messages=5, wait_time_seconds=10
        )

        assert len(messages) == 2
        assert messages[0]["MessageId"] == "msg-1"
        assert messages[1]["MessageId"] == "msg-2"

        mock_sqs_client.receive_message.assert_called_once_with(
            QueueUrl="https://sqs.us-east-1.amazonaws.com/123456789012/test-health-queue",
            MaxNumberOfMessages=5,
            WaitTimeSeconds=10,
            MessageAttributeNames=["All"],
        )

    @pytest.mark.asyncio
    async def test_receive_messages_empty_queue(
        self, aws_messaging_service: AWSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test receiving from empty queue."""
        mock_sqs_client.receive_message.return_value = {}

        messages = await aws_messaging_service.receive_messages("test-health-queue")

        assert messages == []

    @pytest.mark.asyncio
    async def test_receive_messages_error(
        self, aws_messaging_service: AWSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test receiving messages with error."""
        mock_sqs_client.receive_message.side_effect = Exception("Receive error")

        with pytest.raises(Exception, match="Receive error"):
            await aws_messaging_service.receive_messages("test-health-queue")


class TestDeleteMessage:
    """Test message deletion functionality."""

    @pytest.mark.asyncio
    async def test_delete_message_success(
        self, aws_messaging_service: AWSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test successful message deletion."""
        await aws_messaging_service.delete_message(
            "test-health-queue", "receipt-handle-123"
        )

        mock_sqs_client.delete_message.assert_called_once_with(
            QueueUrl="https://sqs.us-east-1.amazonaws.com/123456789012/test-health-queue",
            ReceiptHandle="receipt-handle-123",
        )

    @pytest.mark.asyncio
    async def test_delete_message_error(
        self, aws_messaging_service: AWSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test message deletion with error."""
        mock_sqs_client.delete_message.side_effect = Exception("Delete error")

        with pytest.raises(Exception, match="Delete error"):
            await aws_messaging_service.delete_message(
                "test-health-queue", "receipt-123"
            )


class TestQueueOperations:
    """Test queue-level operations."""

    @pytest.mark.asyncio
    async def test_get_queue_attributes_success(
        self, aws_messaging_service: AWSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test getting queue attributes."""
        mock_sqs_client.get_queue_attributes.return_value = {
            "Attributes": {
                "ApproximateNumberOfMessages": "10",
                "ApproximateNumberOfMessagesNotVisible": "5",
            }
        }

        attributes = await aws_messaging_service.get_queue_attributes(
            "test-health-queue"
        )

        assert attributes["ApproximateNumberOfMessages"] == "10"
        assert attributes["ApproximateNumberOfMessagesNotVisible"] == "5"

        mock_sqs_client.get_queue_attributes.assert_called_once_with(
            QueueUrl="https://sqs.us-east-1.amazonaws.com/123456789012/test-health-queue",
            AttributeNames=["All"],
        )

    @pytest.mark.asyncio
    async def test_get_queue_attributes_error(
        self, aws_messaging_service: AWSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test getting queue attributes with error."""
        mock_sqs_client.get_queue_attributes.side_effect = Exception("Attributes error")

        with pytest.raises(Exception, match="Attributes error"):
            await aws_messaging_service.get_queue_attributes("test-health-queue")

    @pytest.mark.asyncio
    async def test_purge_queue_success(
        self, aws_messaging_service: AWSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test purging queue."""
        await aws_messaging_service.purge_queue("test-health-queue")

        mock_sqs_client.purge_queue.assert_called_once_with(
            QueueUrl="https://sqs.us-east-1.amazonaws.com/123456789012/test-health-queue"
        )

    @pytest.mark.asyncio
    async def test_purge_queue_error(
        self, aws_messaging_service: AWSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test purging queue with error."""
        mock_sqs_client.purge_queue.side_effect = Exception("Purge error")

        with pytest.raises(Exception, match="Purge error"):
            await aws_messaging_service.purge_queue("test-health-queue")


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_success(
        self, aws_messaging_service: AWSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test successful health check."""
        mock_sqs_client.get_queue_attributes.return_value = {
            "Attributes": {"ApproximateNumberOfMessages": "5"}
        }

        result = await aws_messaging_service.health_check()

        assert result["status"] == "healthy"
        assert result["region"] == "us-east-1"
        assert result["sns_configured"] is True
        assert "queues" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_health_check_failure(
        self, aws_messaging_service: AWSMessagingService, mock_sqs_client: MagicMock
    ) -> None:
        """Test health check with failure."""
        mock_sqs_client.get_queue_attributes.side_effect = Exception(
            "Health check failed"
        )

        result = await aws_messaging_service.health_check()

        assert result["status"] == "unhealthy"
        assert "error" in result
        assert "timestamp" in result


class TestCloseAndSingleton:
    """Test close method and singleton pattern."""

    def test_close(self, aws_messaging_service: AWSMessagingService) -> None:
        """Test close method."""
        # Should not raise
        aws_messaging_service.close()

    def test_get_messaging_service_singleton(self) -> None:
        """Test get_messaging_service singleton pattern."""
        with (
            patch("clarity.services.aws_messaging_service._messaging_service", None),
            patch("boto3.client"),
        ):
            service1 = get_messaging_service()
            service2 = get_messaging_service()

            assert service1 is service2

    def test_get_messaging_service_with_params(self) -> None:
        """Test get_messaging_service with custom params."""
        with (
            patch("clarity.services.aws_messaging_service._messaging_service", None),
            patch("boto3.client"),
        ):
            service = get_messaging_service(
                region="eu-west-1",
                health_data_queue="custom-health-queue",
                insight_queue="custom-insight-queue",
                sns_topic_arn="arn:aws:sns:eu-west-1:123456789012:custom-topic",
            )

            assert service.region == "eu-west-1"
            assert service.health_data_queue == "custom-health-queue"
            assert service.insight_queue == "custom-insight-queue"
            assert (
                service.sns_topic_arn
                == "arn:aws:sns:eu-west-1:123456789012:custom-topic"
            )


class TestPydanticModels:
    """Test Pydantic models."""

    def test_health_data_event(self) -> None:
        """Test HealthDataEvent model."""
        event = HealthDataEvent(
            user_id="user-123",
            upload_id="upload-456",
            s3_path="s3://bucket/path.json",
            timestamp="2024-01-15T12:00:00Z",
            metadata={"source": "mobile"},
        )

        assert event.user_id == "user-123"
        assert event.upload_id == "upload-456"
        assert event.s3_path == "s3://bucket/path.json"
        assert event.event_type == "health_data_upload"
        assert event.metadata == {"source": "mobile"}

    def test_insight_request_event(self) -> None:
        """Test InsightRequestEvent model."""
        event = InsightRequestEvent(
            user_id="user-123",
            upload_id="upload-456",
            analysis_results={"risk_score": 0.5},
            timestamp="2024-01-15T12:00:00Z",
            metadata={"version": "1.0"},
        )

        assert event.user_id == "user-123"
        assert event.upload_id == "upload-456"
        assert event.analysis_results == {"risk_score": 0.5}
        assert event.event_type == "insight_request"
        assert event.metadata == {"version": "1.0"}
