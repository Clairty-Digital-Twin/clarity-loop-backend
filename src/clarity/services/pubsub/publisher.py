"""Health Data Publisher Service.

Publishes health data processing events to Pub/Sub topics for async processing.
"""

from datetime import datetime
import json
import logging
import os
from typing import Any

from google.cloud import pubsub_v1
from pydantic import BaseModel

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
    """Publisher service for health data events."""

    def __init__(self) -> None:
        """Initialize publisher."""
        self.logger = logging.getLogger(__name__)
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

        # Initialize publisher client
        self.publisher = pubsub_v1.PublisherClient()

        # Topic names
        self.health_data_topic = os.getenv("PUBSUB_HEALTH_DATA_TOPIC", "health-data-upload")
        self.insight_request_topic = os.getenv("PUBSUB_INSIGHTS_TOPIC", "insight-request")

        # Topic paths
        self.health_data_topic_path = self.publisher.topic_path(self.project_id, self.health_data_topic)
        self.insight_request_topic_path = self.publisher.topic_path(self.project_id, self.insight_request_topic)

        self.logger.info("Initialized health data publisher for project: %s", self.project_id)

    async def publish_health_data_event(
        self,
        user_id: str,
        upload_id: str,
        gcs_path: str,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Publish health data upload event.
        
        Args:
            user_id: User identifier
            upload_id: Unique upload identifier
            gcs_path: Path to raw data in GCS
            metadata: Optional metadata
            
        Returns:
            Message ID from Pub/Sub
        """
        try:
            event = HealthDataEvent(
                user_id=user_id,
                upload_id=upload_id,
                gcs_path=gcs_path,
                timestamp=datetime.utcnow().isoformat(),
                metadata=metadata or {}
            )

            # Serialize event to JSON
            message_data = event.json().encode("utf-8")

            # Add message attributes
            attributes = {
                "user_id": user_id,
                "upload_id": upload_id,
                "event_type": "health_data_upload"
            }

            # Publish message
            future = self.publisher.publish(
                self.health_data_topic_path,
                message_data,
                **attributes
            )

            message_id = future.result()  # Wait for publish to complete

            self.logger.info("Published health data event: user=%s, upload=%s, message_id=%s",
                           user_id, upload_id, message_id)

            return message_id

        except Exception as e:
            self.logger.error("Failed to publish health data event: %s", e)
            raise

    async def publish_insight_request_event(
        self,
        user_id: str,
        upload_id: str,
        analysis_results: dict[str, Any],
        metadata: dict[str, Any] | None = None
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
            event = InsightRequestEvent(
                user_id=user_id,
                upload_id=upload_id,
                analysis_results=analysis_results,
                timestamp=datetime.utcnow().isoformat(),
                metadata=metadata or {}
            )

            # Serialize event to JSON
            message_data = event.json().encode("utf-8")

            # Add message attributes
            attributes = {
                "user_id": user_id,
                "upload_id": upload_id,
                "event_type": "insight_request"
            }

            # Publish message
            future = self.publisher.publish(
                self.insight_request_topic_path,
                message_data,
                **attributes
            )

            message_id = future.result()  # Wait for publish to complete

            self.logger.info("Published insight request event: user=%s, upload=%s, message_id=%s",
                           user_id, upload_id, message_id)

            return message_id

        except Exception as e:
            self.logger.error("Failed to publish insight request event: %s", e)
            raise

    def close(self) -> None:
        """Close publisher client."""
        if hasattr(self.publisher, 'close'):
            self.publisher.close()


# Global publisher instance
_publisher: HealthDataPublisher | None = None


def get_publisher() -> HealthDataPublisher:
    """Get or create global publisher instance."""
    global _publisher

    if _publisher is None:
        _publisher = HealthDataPublisher()

    return _publisher
