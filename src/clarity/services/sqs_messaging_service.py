"""AWS SQS messaging service for asynchronous processing."""

# removed - breaks FastAPI

from datetime import UTC, datetime
import json
import logging
from typing import Any, cast
import uuid

import boto3
from botocore.exceptions import ClientError

from clarity.core.exceptions import ServiceError


# Create a specific messaging error type
class MessagingError(ServiceError):
    """Raised when messaging operations fail."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_code="MESSAGING_ERROR", details=kwargs)


logger = logging.getLogger(__name__)


class SQSMessagingService:
    """AWS SQS service for async message processing."""

    def __init__(
        self,
        queue_url: str,
        region: str = "us-east-1",
        sns_topic_arn: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        self.queue_url = queue_url
        self.sns_topic_arn = sns_topic_arn
        self.region = region

        # Create SQS client
        if endpoint_url:  # For local testing with LocalStack
            self.sqs_client = boto3.client(
                "sqs", region_name=region, endpoint_url=endpoint_url
            )
            if sns_topic_arn:
                self.sns_client = boto3.client(
                    "sns", region_name=region, endpoint_url=endpoint_url
                )
        else:
            self.sqs_client = boto3.client("sqs", region_name=region)
            if sns_topic_arn:
                self.sns_client = boto3.client("sns", region_name=region)

    async def publish_message(
        self,
        message_type: str,
        data: dict[str, Any],
        attributes: dict[str, str] | None = None,
    ) -> str:
        """Publish message to SQS queue."""
        try:
            # Prepare message
            message_id = str(uuid.uuid4())
            message_body = {
                "id": message_id,
                "type": message_type,
                "timestamp": datetime.now(UTC).isoformat(),
                "data": data,
            }

            # Prepare message attributes
            message_attributes = {
                "MessageType": {"DataType": "String", "StringValue": message_type}
            }

            if attributes:
                for key, value in attributes.items():
                    message_attributes[key] = {
                        "DataType": "String",
                        "StringValue": str(value),
                    }

            # Send to SQS
            response = self.sqs_client.send_message(
                QueueUrl=self.queue_url,
                MessageBody=json.dumps(message_body),
                MessageAttributes=message_attributes,  # type: ignore[arg-type]
            )

            logger.info("Published message %s to SQS", message_id)
            return response["MessageId"]

        except ClientError as e:
            logger.exception("SQS publish error")
            msg = f"Failed to publish message: {e!s}"
            raise MessagingError(msg) from e
        except Exception as e:
            logger.exception("Unexpected error publishing message")
            msg = f"Failed to publish message: {e!s}"
            raise MessagingError(msg) from e

    async def receive_messages(
        self,
        max_messages: int = 10,
        wait_time_seconds: int = 20,
        visibility_timeout: int = 30,
    ) -> list[dict[str, Any]]:
        """Receive messages from SQS queue."""
        try:
            response = self.sqs_client.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=max_messages,
                WaitTimeSeconds=wait_time_seconds,
                VisibilityTimeout=visibility_timeout,
                MessageAttributeNames=["All"],
                AttributeNames=["All"],
            )

            messages = []
            for msg in response.get("Messages", []):
                try:
                    body = json.loads(msg["Body"])
                    messages.append(
                        {
                            "receipt_handle": msg["ReceiptHandle"],
                            "message_id": msg["MessageId"],
                            "body": body,
                            "attributes": msg.get("MessageAttributes", {}),
                            "system_attributes": msg.get("Attributes", {}),
                        }
                    )
                except json.JSONDecodeError:
                    logger.exception("Failed to decode message: %s", msg["MessageId"])
                    continue

            return messages

        except ClientError as e:
            logger.exception("SQS receive error")
            error_msg = f"Failed to receive messages: {e!s}"
            raise MessagingError(error_msg) from e

    async def delete_message(self, receipt_handle: str) -> None:
        """Delete message from SQS queue."""
        try:
            self.sqs_client.delete_message(
                QueueUrl=self.queue_url, ReceiptHandle=receipt_handle
            )

            logger.info("Successfully deleted message from SQS")

        except ClientError as e:
            logger.exception("SQS delete error")
            msg = f"Failed to delete message: {e!s}"
            raise MessagingError(msg) from e

    async def batch_delete_messages(self, receipt_handles: list[str]) -> dict[str, Any]:
        """Batch delete messages from SQS."""
        try:
            entries = [
                {"Id": str(i), "ReceiptHandle": handle}
                for i, handle in enumerate(receipt_handles)
            ]

            response = self.sqs_client.delete_message_batch(
                QueueUrl=self.queue_url,
                Entries=entries,  # type: ignore[arg-type]
            )

            return {
                "successful": response.get("Successful", []),
                "failed": response.get("Failed", []),
            }

        except ClientError as e:
            logger.exception("SQS batch delete error")
            msg = f"Failed to batch delete messages: {e!s}"
            raise MessagingError(msg) from e

    async def publish_to_sns(
        self,
        subject: str,
        message: dict[str, Any],
        attributes: dict[str, str] | None = None,
    ) -> str:
        """Publish message to SNS topic for fan-out."""
        if not self.sns_topic_arn:
            msg = "SNS topic ARN not configured"
            raise MessagingError(msg)

        try:
            # Prepare message attributes
            message_attributes = {}
            if attributes:
                for key, value in attributes.items():
                    message_attributes[key] = {
                        "DataType": "String",
                        "StringValue": str(value),
                    }

            # Publish to SNS
            response = self.sns_client.publish(
                TopicArn=self.sns_topic_arn,
                Subject=subject,
                Message=json.dumps(message),
                MessageAttributes=message_attributes,
            )

            logger.info("Published message to SNS: %s", response["MessageId"])
            return str(response["MessageId"])

        except ClientError as e:
            logger.exception("SNS publish error")
            msg = f"Failed to publish to SNS: {e!s}"
            raise MessagingError(msg) from e

    async def get_queue_attributes(self) -> dict[str, Any]:
        """Get queue attributes and statistics."""
        try:
            response = self.sqs_client.get_queue_attributes(
                QueueUrl=self.queue_url, AttributeNames=["All"]
            )

            return cast("dict[str, Any]", response.get("Attributes", {}))

        except ClientError as e:
            logger.exception("SQS get attributes error")
            msg = f"Failed to get queue attributes: {e!s}"
            raise MessagingError(msg) from e

    async def purge_queue(self) -> None:
        """Purge all messages from queue (use with caution)."""
        try:
            self.sqs_client.purge_queue(QueueUrl=self.queue_url)
            logger.warning("Purged all messages from queue: %s", self.queue_url)

        except ClientError as e:
            logger.exception("SQS purge error")
            msg = f"Failed to purge queue: {e!s}"
            raise MessagingError(msg) from e


class HealthDataMessageTypes:
    """Message types for health data processing."""

    HEALTH_DATA_UPLOADED = "health_data_uploaded"
    ANALYSIS_REQUESTED = "analysis_requested"
    ANALYSIS_COMPLETED = "analysis_completed"
    INSIGHT_GENERATED = "insight_generated"
    ERROR_OCCURRED = "error_occurred"
