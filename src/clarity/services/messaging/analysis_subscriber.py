"""Analysis Subscriber Service.

Handles Pub/Sub messages for health data analysis processing.
"""

# removed - breaks FastAPI

import base64
import json
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage

from clarity.ml.analysis_pipeline import run_analysis_pipeline
from clarity.services.messaging.publisher import HealthDataPublisher, get_publisher

logger = logging.getLogger(__name__)


class AnalysisSubscriber:
    """Subscriber service for health data analysis."""

    def __init__(self) -> None:
        """Initialize analysis subscriber."""
        self.logger = logging.getLogger(__name__)
        self.storage_client = storage.Client()
        self.publisher: HealthDataPublisher | None = (
            None  # Will be initialized asynchronously
        )

        # Environment settings
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.pubsub_push_audience = os.getenv("PUBSUB_PUSH_AUDIENCE")

        self.logger.info("Initialized analysis subscriber (env: %s)", self.environment)

    async def _get_publisher(self) -> HealthDataPublisher:
        """Get or initialize the publisher asynchronously."""
        if self.publisher is None:
            self.publisher = await get_publisher()
        return self.publisher

    async def process_health_data_message(self, request: Request) -> dict[str, Any]:
        """Process incoming Pub/Sub message for health data analysis.

        Args:
            request: FastAPI request object containing Pub/Sub message

        Returns:
            Processing result
        """
        try:
            # Verify Pub/Sub authentication in production
            if self.environment == "production":
                await self._verify_pubsub_token(request)

            # Parse Pub/Sub message
            body = await request.json()
            message_data = self._extract_message_data(body)

            self.logger.info(
                "Processing health data analysis for user: %s, upload: %s",
                message_data.get("user_id"),
                message_data.get("upload_id"),
            )

            # Download raw data from GCS
            raw_health_data = await self._download_health_data(message_data["gcs_path"])

            # Run analysis pipeline
            analysis_results = await run_analysis_pipeline(
                user_id=message_data["user_id"], health_data=raw_health_data
            )

            # Publish insight request event
            publisher = await self._get_publisher()
            await publisher.publish_insight_request(
                user_id=message_data["user_id"],
                upload_id=message_data["upload_id"],
                analysis_results=analysis_results,
            )

            self.logger.info(
                "Completed health data analysis for user: %s", message_data["user_id"]
            )

            return {
                "status": "success",
                "user_id": message_data["user_id"],
                "upload_id": message_data["upload_id"],
                "analysis_completed": True,
            }

        except Exception as e:
            self.logger.exception("Error processing health data message")
            raise HTTPException(
                status_code=500, detail=f"Analysis processing failed: {e!s}"
            ) from e

    async def _verify_pubsub_token(self, request: Request) -> None:
        """Verify Pub/Sub OIDC token in production."""
        authorization = request.headers.get("authorization")

        if not authorization:
            raise HTTPException(status_code=401, detail="Missing authorization header")

        try:
            # Extract token from "Bearer <token>" format
            token = (
                authorization.split(" ")[1] if " " in authorization else authorization
            )

            # TODO: Implement proper JWT verification using Google's public keys
            # For now, just check that token exists
            if not token:
                self._raise_invalid_token_error()

            # In production, you would verify the JWT signature and claims here
            # using google.auth.jwt or similar library

        except Exception as e:
            self.logger.exception("Token verification failed")
            raise HTTPException(status_code=401, detail="Invalid Pub/Sub token") from e

    def _extract_message_data(self, pubsub_body: dict[str, Any]) -> dict[str, Any]:
        """Extract and decode Pub/Sub message data."""
        try:
            # Pub/Sub push format: {"message": {"data": "<base64>", "attributes": {...}}}
            message = pubsub_body.get("message", {})

            # Decode base64 data
            encoded_data = message.get("data", "")
            decoded_data = base64.b64decode(encoded_data).decode("utf-8")

            # Parse JSON
            message_data = json.loads(decoded_data)

            # Validate required fields
            required_fields = ["user_id", "upload_id", "gcs_path"]
            for field in required_fields:
                if field not in message_data:
                    self._raise_missing_field_error(field)

        except Exception as e:
            self.logger.exception("Failed to extract message data")
            raise HTTPException(
                status_code=400, detail=f"Invalid message format: {e!s}"
            ) from e
        else:
            return message_data  # type: ignore[no-any-return]

    async def _download_health_data(self, gcs_path: str) -> dict[str, Any]:
        """Download raw health data from GCS.

        Args:
            gcs_path: GCS path in format gs://bucket/path

        Returns:
            Raw health data as dictionary
        """
        try:
            # Parse GCS path
            if not gcs_path.startswith("gs://"):
                self._raise_invalid_gcs_path_error(gcs_path)

            path_parts = gcs_path[5:].split("/", 1)  # Remove "gs://" prefix
            bucket_name = path_parts[0]
            blob_path = path_parts[1] if len(path_parts) > 1 else ""

            # Download data
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            if not blob.exists():
                self._raise_health_data_not_found_error(gcs_path)

            # Download and parse JSON
            raw_json = blob.download_as_text()
            health_data = json.loads(raw_json)

            self.logger.info(
                "Downloaded health data from GCS: %s (%d bytes)",
                gcs_path,
                len(raw_json),
            )

        except Exception:
            self.logger.exception("Failed to download health data from %s", gcs_path)
            raise
        else:
            return health_data  # type: ignore[no-any-return]

    @staticmethod
    def _raise_invalid_token_error() -> None:
        """Raise HTTPException for invalid token format."""
        raise HTTPException(status_code=401, detail="Invalid token format")

    @staticmethod
    def _raise_missing_field_error(field: str) -> None:
        """Raise ValueError for missing required field."""
        msg = f"Missing required field: {field}"
        raise ValueError(msg)

    @staticmethod
    def _raise_invalid_gcs_path_error(gcs_path: str) -> None:
        """Raise ValueError for invalid GCS path format."""
        msg = f"Invalid GCS path format: {gcs_path}"
        raise ValueError(msg)

    @staticmethod
    def _raise_health_data_not_found_error(gcs_path: str) -> None:
        """Raise FileNotFoundError for missing health data."""
        msg = f"Health data not found at: {gcs_path}"
        raise FileNotFoundError(msg)


# Create FastAPI app for analysis service
analysis_app = FastAPI(title="Health Data Analysis Service")


@analysis_app.post("/process-task")
async def process_analysis_task(request: Request) -> dict[str, Any]:
    """Handle Pub/Sub push subscription for health data analysis."""
    subscriber = get_analysis_subscriber()
    return await subscriber.process_health_data_message(request)


@analysis_app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "analysis"}


class AnalysisSubscriberSingleton:
    """Singleton container for analysis subscriber."""

    _instance: AnalysisSubscriber | None = None

    @classmethod
    def get_instance(cls) -> AnalysisSubscriber:
        """Get or create analysis subscriber instance."""
        if cls._instance is None:
            cls._instance = AnalysisSubscriber()
        return cls._instance


def get_analysis_subscriber() -> AnalysisSubscriber:
    """Get or create global analysis subscriber instance."""
    return AnalysisSubscriberSingleton.get_instance()
