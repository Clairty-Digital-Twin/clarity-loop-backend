"""Health Data Publisher Service - AWS SQS/SNS Edition.

Publishes health data processing events to AWS SQS/SNS for async processing.
Replaces Google Pub/Sub with enterprise-grade AWS messaging.
"""

# removed - breaks FastAPI

import logging
import os
from typing import Any

from pydantic import BaseModel

from clarity.core.decorators import log_execution
from clarity.services.aws_messaging_service import AWSMessagingService

logger = logging.getLogger(__name__)


class HealthDataEvent(BaseModel):
    """Health data processing event."""

    user_id: str
    upload_id: str
    s3_path: str  # Changed from gcs_path to s3_path
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


class HealthDataPublisher:
    """AWS-powered publisher for health data processing events."""

    def __init__(self) -> None:
        """Initialize the publisher with AWS SQS/SNS messaging."""
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.sns_topic_arn = os.getenv("CLARITY_SNS_TOPIC_ARN")

        # Initialize AWS messaging service
        self.messaging_service = AWSMessagingService(
            region=self.aws_region,
            health_data_queue=os.getenv(
                "CLARITY_HEALTH_DATA_QUEUE", "clarity-health-data-processing"
            ),
            insight_queue=os.getenv(
                "CLARITY_INSIGHT_QUEUE", "clarity-insight-generation"
            ),
            sns_topic_arn=self.sns_topic_arn,
        )

        self.logger = logging.getLogger(__name__)

        self.logger.info(
            "Initialized HealthDataPublisher for AWS region: %s", self.aws_region
        )

    @log_execution(level=logging.DEBUG)
    async def publish_health_data_upload(
        self,
        user_id: str,
        upload_id: str,
        s3_path: str,  # Changed from gcs_path to s3_path
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Publish health data upload event to AWS SQS.

        Args:
            user_id: User identifier
            upload_id: Upload identifier
            s3_path: Path to raw data in S3
            metadata: Optional metadata

        Returns:
            Message ID from SQS
        """
        try:
            message_id = await self.messaging_service.publish_health_data_upload(
                user_id=user_id,
                upload_id=upload_id,
                s3_path=s3_path,
                metadata=metadata,
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
        """Publish insight generation request event to AWS SQS.

        Args:
            user_id: User identifier
            upload_id: Upload identifier
            analysis_results: Results from analysis pipeline
            metadata: Optional metadata

        Returns:
            Message ID from SQS
        """
        try:
            message_id = await self.messaging_service.publish_insight_request(
                user_id=user_id,
                upload_id=upload_id,
                analysis_results=analysis_results,
                metadata=metadata,
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

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on AWS messaging services."""
        return await self.messaging_service.health_check()

    def close(self) -> None:
        """Close publisher connections."""
        self.messaging_service.close()


# Global singleton instance
_publisher: HealthDataPublisher | None = None


async def get_publisher() -> HealthDataPublisher:  # noqa: RUF029
    """Get or create global publisher instance."""
    global _publisher  # noqa: PLW0603 - Singleton pattern for messaging publisher

    if _publisher is None:
        _publisher = HealthDataPublisher()

    return _publisher
