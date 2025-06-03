"""Health Data Publisher Service.

Publishes health data processing events to Pub/Sub topics for async processing.
"""

from datetime import UTC, datetime
import json
import logging
import os
from typing import Any

from google.cloud.pubsub_v1 import PublisherClient
from pydantic import BaseModel

from clarity.core.decorators import log_execution

logger = logging.getLogger(__name__)


class HealthDataEvent(BaseModel):
    """Health data processing event."""

    user_id: str
    upload_id: str
    gcs_path: str
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
    """Publisher for health data processing events."""

    def __init__(self) -> None:
        """Initialize the publisher with GCP Pub/Sub client."""
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.publisher = PublisherClient()
        self.logger = logging.getLogger(__name__)

        # Topic names
        self.health_data_topic = (
            f"projects/{self.project_id}/topics/health-data-uploads"
        )
        self.insight_topic = f"projects/{self.project_id}/topics/insight-requests"

        self.logger.info(
            "Initialized HealthDataPublisher for project: %s", self.project_id
        )

    @log_execution(level=logging.DEBUG)
    def publish_health_data_upload(
        self,
        user_id: str,
        upload_id: str,
        gcs_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Publish health data upload event.

        Args:
            user_id: User identifier
            upload_id: Upload identifier
            gcs_path: Path to raw data in GCS
            metadata: Optional metadata

        Returns:
            Message ID from Pub/Sub
        """
        try:
            # Create message payload
            message_data = {
                "user_id": user_id,
                "upload_id": upload_id,
                "gcs_path": gcs_path,
                "timestamp": datetime.now(UTC).isoformat(),
                "metadata": metadata or {},
            }

            # Convert to JSON bytes
            message_bytes = json.dumps(message_data).encode("utf-8")

            # Publish with retry
            future = self.publisher.publish(
                self.health_data_topic,
                message_bytes,
                user_id=user_id,
                upload_id=upload_id,
            )

            # Get message ID
            message_id: str = future.result(timeout=30.0)

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
    def publish_insight_request(
        self,
        user_id: str,
        upload_id: str,
        analysis_results: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Publish insight generation request event.

        Args:
            user_id: User identifier
            upload_id: Upload identifier
            analysis_results: Results from analysis pipeline
            metadata: Optional metadata

        Returns:
            Message ID from Pub/Sub
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

            # Convert to JSON bytes
            message_bytes = json.dumps(message_data).encode("utf-8")

            # Publish with retry
            future = self.publisher.publish(
                self.insight_topic,
                message_bytes,
                user_id=user_id,
                upload_id=upload_id,
            )

            # Get message ID
            message_id: str = future.result(timeout=30.0)

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

    def close(self) -> None:
        """Close publisher client."""
        # PublisherClient doesn't have a close method in current version
        # Client connections are automatically managed


# Global singleton instance
_publisher: HealthDataPublisher | None = None


def get_publisher() -> HealthDataPublisher:
    """Get or create global publisher instance."""
    global _publisher  # noqa: PLW0603

    if _publisher is None:
        _publisher = HealthDataPublisher()

    return _publisher
