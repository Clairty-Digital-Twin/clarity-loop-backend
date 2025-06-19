"""CLARITY Digital Twin Platform - AWS SQS/SNS Messaging Service.

Enterprise-grade AWS messaging service for async health data processing.
Replaces Google Pub/Sub with AWS-native solution using SQS and SNS.
"""

# removed - breaks FastAPI

import asyncio
from datetime import UTC, datetime
import json
import logging
from typing import Any, cast

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel

from clarity.core.decorators import log_execution

logger = logging.getLogger(__name__)


class HealthDataEvent(BaseModel):
    """Health data processing event."""

    user_id: str
    upload_id: str
    s3_path: str
    event_type: str = "health_data_upload"
    timestamp: str
    metadata: dict[str, Any] = {}


class InsightRequestEvent(BaseModel):
    """Insight generation request event."""

    user_id: str
    upload_id: str
    analysis_results: dict[str, Any]
    event_type: str = "insight_request"
    timestamp: str
    metadata: dict[str, Any] = {}


class AWSMessagingService:
    """AWS SQS/SNS messaging service for health data processing events.

    Replaces Google Pub/Sub with enterprise-grade AWS messaging.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
        health_data_queue: str = "clarity-health-data-processing",
        insight_queue: str = "clarity-insight-generation",
        sns_topic_arn: str | None = None,
    ) -> None:
        """Initialize AWS messaging service.

        Args:
            region: AWS region
            endpoint_url: Optional endpoint URL (for local testing)
            health_data_queue: SQS queue name for health data processing
            insight_queue: SQS queue name for insight generation
            sns_topic_arn: Optional SNS topic ARN for fan-out messaging
        """
        self.region = region
        self.endpoint_url = endpoint_url
        self.health_data_queue = health_data_queue
        self.insight_queue = insight_queue
        self.sns_topic_arn = sns_topic_arn

        # Initialize AWS clients
        self.sqs_client = boto3.client(
            "sqs",
            region_name=region,
            endpoint_url=endpoint_url,
        )

        if sns_topic_arn:
            self.sns_client = boto3.client(
                "sns",
                region_name=region,
                endpoint_url=endpoint_url,
            )
        else:
            self.sns_client = None

        self.logger = logging.getLogger(__name__)

        # Get queue URLs (create if they don't exist)
        self._queue_urls: dict[str, str] = {}

        logger.info("AWS messaging service initialized for region: %s", region)

    async def _get_queue_url(self, queue_name: str) -> str:
        """Get or create SQS queue URL."""
        if queue_name in self._queue_urls:
            return self._queue_urls[queue_name]

        loop = asyncio.get_event_loop()
        try:
            # Try to get existing queue
            response = await loop.run_in_executor(
                None, lambda: self.sqs_client.get_queue_url(QueueName=queue_name)
            )
            queue_url = response["QueueUrl"]

        except ClientError as e:
            if e.response["Error"]["Code"] == "AWS.SimpleQueueService.NonExistentQueue":
                # Create the queue if it doesn't exist
                logger.info("Creating SQS queue: %s", queue_name)
                response = await loop.run_in_executor(
                    None,
                    lambda: self.sqs_client.create_queue(
                        QueueName=queue_name,
                        Attributes={
                            "MessageRetentionPeriod": "1209600",  # 14 days
                            "VisibilityTimeout": "300",  # 5 minutes
                            "ReceiveMessageWaitTimeSeconds": "20",  # Long polling
                        },
                    ),
                )
                queue_url = response["QueueUrl"]
            else:
                raise

        self._queue_urls[queue_name] = queue_url
        logger.info("Using SQS queue: %s -> %s", queue_name, queue_url)
        return queue_url

    @log_execution(level=logging.DEBUG)
    async def publish_health_data_upload(
        self,
        user_id: str,
        upload_id: str,
        s3_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Publish health data upload event to SQS.

        Args:
            user_id: User identifier
            upload_id: Upload identifier
            s3_path: Path to raw data in S3
            metadata: Optional metadata

        Returns:
            Message ID from SQS
        """
        try:
            # Create message payload
            message_data = {
                "user_id": user_id,
                "upload_id": upload_id,
                "s3_path": s3_path,
                "timestamp": datetime.now(UTC).isoformat(),
                "metadata": metadata or {},
            }

            # Get queue URL
            queue_url = await self._get_queue_url(self.health_data_queue)

            # Send message to SQS
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.sqs_client.send_message(
                    QueueUrl=queue_url,
                    MessageBody=json.dumps(message_data),
                    MessageAttributes={
                        "user_id": {
                            "StringValue": user_id,
                            "DataType": "String",
                        },
                        "upload_id": {
                            "StringValue": upload_id,
                            "DataType": "String",
                        },
                        "event_type": {
                            "StringValue": "health_data_upload",
                            "DataType": "String",
                        },
                    },
                ),
            )

            message_id: str = response["MessageId"]

            # Also publish to SNS for fan-out if configured
            if self.sns_client and self.sns_topic_arn:
                await self._publish_to_sns(
                    subject="Health Data Upload",
                    message=message_data,
                    attributes={
                        "event_type": "health_data_upload",
                        "user_id": user_id,
                    },
                )

        except Exception:
            self.logger.exception("Failed to publish health data event")
            raise
        else:
            self.logger.info(
                "Published health data upload event for user %s, upload %s, message ID: %s",
                user_id,
                upload_id,
                message_id,
            )
            return message_id

    @log_execution(level=logging.DEBUG)
    async def publish_insight_request(
        self,
        user_id: str,
        upload_id: str,
        analysis_results: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Publish insight generation request event to SQS.

        Args:
            user_id: User identifier
            upload_id: Upload identifier
            analysis_results: Results from analysis pipeline
            metadata: Optional metadata

        Returns:
            Message ID from SQS
        """
        try:
            # Create message payload
            message_data = {
                "user_id": user_id,
                "upload_id": upload_id,
                "analysis_results": analysis_results,
                "timestamp": datetime.now(UTC).isoformat(),
                "metadata": metadata or {},
            }

            # Get queue URL
            queue_url = await self._get_queue_url(self.insight_queue)

            # Send message to SQS
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.sqs_client.send_message(
                    QueueUrl=queue_url,
                    MessageBody=json.dumps(message_data),
                    MessageAttributes={
                        "user_id": {
                            "StringValue": user_id,
                            "DataType": "String",
                        },
                        "upload_id": {
                            "StringValue": upload_id,
                            "DataType": "String",
                        },
                        "event_type": {
                            "StringValue": "insight_request",
                            "DataType": "String",
                        },
                    },
                ),
            )

            message_id: str = response["MessageId"]

            # Also publish to SNS for fan-out if configured
            if self.sns_client and self.sns_topic_arn:
                await self._publish_to_sns(
                    subject="Insight Generation Request",
                    message=message_data,
                    attributes={
                        "event_type": "insight_request",
                        "user_id": user_id,
                    },
                )

        except Exception:
            self.logger.exception("Failed to publish insight request event")
            raise
        else:
            self.logger.info(
                "Published insight request event for user %s, upload %s, message ID: %s",
                user_id,
                upload_id,
                message_id,
            )
            return message_id

    async def _publish_to_sns(
        self,
        subject: str,
        message: dict[str, Any],
        attributes: dict[str, str] | None = None,
    ) -> str:
        """Publish message to SNS topic for fan-out.

        Args:
            subject: Message subject
            message: Message content
            attributes: Message attributes

        Returns:
            SNS message ID
        """
        if not self.sns_client or not self.sns_topic_arn:
            msg = "SNS not configured"
            raise RuntimeError(msg)

        try:
            # Prepare SNS attributes
            sns_attributes = {}
            if attributes:
                for key, value in attributes.items():
                    sns_attributes[key] = {
                        "DataType": "String",
                        "StringValue": value,
                    }

            # Publish to SNS
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.sns_client.publish(
                    TopicArn=self.sns_topic_arn,
                    Subject=subject,
                    Message=json.dumps(message, indent=2),
                    MessageAttributes=sns_attributes,
                ),
            )

            message_id: str = response["MessageId"]
            logger.info("Published message to SNS: %s", message_id)

        except Exception:
            logger.exception("Failed to publish to SNS")
            raise
        else:
            return message_id

    async def receive_messages(
        self,
        queue_name: str,
        max_messages: int = 10,
        wait_time_seconds: int = 20,
    ) -> list[dict[str, Any]]:
        """Receive messages from SQS queue.

        Args:
            queue_name: Queue name to receive from
            max_messages: Maximum messages to receive
            wait_time_seconds: Long polling wait time

        Returns:
            List of message dictionaries
        """
        try:
            queue_url = await self._get_queue_url(queue_name)

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.sqs_client.receive_message(
                    QueueUrl=queue_url,
                    MaxNumberOfMessages=max_messages,
                    WaitTimeSeconds=wait_time_seconds,
                    MessageAttributeNames=["All"],
                ),
            )

            messages = response.get("Messages", [])
            logger.info("Received %d messages from queue %s", len(messages), queue_name)

        except Exception:
            logger.exception("Failed to receive messages from queue %s", queue_name)
            raise
        else:
            # Cast AWS MessageTypeDef to dict[str, Any] for our interface
            return [dict(msg) for msg in messages]

    async def delete_message(self, queue_name: str, receipt_handle: str) -> None:
        """Delete processed message from SQS queue.

        Args:
            queue_name: Queue name
            receipt_handle: Message receipt handle
        """
        try:
            queue_url = await self._get_queue_url(queue_name)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.sqs_client.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=receipt_handle,
                ),
            )

            logger.debug("Deleted message from queue %s", queue_name)

        except Exception:
            logger.exception("Failed to delete message from queue %s", queue_name)
            raise

    async def get_queue_attributes(self, queue_name: str) -> dict[str, Any]:
        """Get queue attributes and metrics.

        Args:
            queue_name: Queue name

        Returns:
            Dictionary of queue attributes
        """
        try:
            queue_url = await self._get_queue_url(queue_name)

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.sqs_client.get_queue_attributes(
                    QueueUrl=queue_url,
                    AttributeNames=["All"],
                ),
            )

            attributes = response.get("Attributes", {})
            logger.debug("Retrieved attributes for queue %s", queue_name)

        except Exception:
            logger.exception("Failed to get attributes for queue %s", queue_name)
            raise
        else:
            # Cast AWS attribute dict to dict[str, Any] for our interface
            return cast("dict[str, Any]", attributes)

    async def purge_queue(self, queue_name: str) -> None:
        """Purge all messages from queue (use with caution).

        Args:
            queue_name: Queue name to purge
        """
        try:
            queue_url = await self._get_queue_url(queue_name)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: self.sqs_client.purge_queue(QueueUrl=queue_url)
            )
            logger.warning("Purged queue %s", queue_name)

        except Exception:
            logger.exception("Failed to purge queue %s", queue_name)
            raise

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on AWS messaging services.

        Returns:
            Health status information
        """
        try:
            # Test SQS connection
            health_data_url = await self._get_queue_url(self.health_data_queue)
            insight_url = await self._get_queue_url(self.insight_queue)

            # Get queue attributes to verify connectivity
            health_attrs = await self.get_queue_attributes(self.health_data_queue)
            insight_attrs = await self.get_queue_attributes(self.insight_queue)

            return {
                "status": "healthy",
                "region": self.region,
                "queues": {
                    "health_data": {
                        "url": health_data_url,
                        "messages_available": health_attrs.get(
                            "ApproximateNumberOfMessages", "0"
                        ),
                    },
                    "insight": {
                        "url": insight_url,
                        "messages_available": insight_attrs.get(
                            "ApproximateNumberOfMessages", "0"
                        ),
                    },
                },
                "sns_configured": bool(self.sns_topic_arn),
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.exception("AWS messaging health check failed")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def close(self) -> None:
        """Close messaging service connections."""
        # AWS SDK clients handle connection pooling automatically
        logger.info("AWS messaging service closed")


# Global singleton instance
_messaging_service: AWSMessagingService | None = None


def get_messaging_service(
    region: str = "us-east-1",
    endpoint_url: str | None = None,
    health_data_queue: str = "clarity-health-data-processing",
    insight_queue: str = "clarity-insight-generation",
    sns_topic_arn: str | None = None,
) -> AWSMessagingService:
    """Get or create global AWS messaging service instance."""
    global _messaging_service  # noqa: PLW0603 - Singleton pattern for AWS messaging service

    if _messaging_service is None:
        _messaging_service = AWSMessagingService(
            region=region,
            endpoint_url=endpoint_url,
            health_data_queue=health_data_queue,
            insight_queue=insight_queue,
            sns_topic_arn=sns_topic_arn,
        )

    return _messaging_service
