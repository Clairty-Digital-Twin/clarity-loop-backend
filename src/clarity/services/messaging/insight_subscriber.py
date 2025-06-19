"""Insight Subscriber Service.

Handles Pub/Sub messages for AI-powered health insight generation using Gemini.
"""

# removed - breaks FastAPI

import base64
import json
import logging
import os
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage

from clarity.ml.gemini_service import GeminiService, HealthInsightRequest

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class InsightSubscriber:
    """Subscriber service for health insight generation."""

    def __init__(self) -> None:
        """Initialize insight subscriber."""
        self.logger = logging.getLogger(__name__)
        self.storage_client = storage.Client()
        self.gemini_service = GeminiService()

        # Environment settings
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.pubsub_push_audience = os.getenv("PUBSUB_PUSH_AUDIENCE")

        self.logger.info("Initialized insight subscriber (env: %s)", self.environment)

    async def process_insight_request(self, request: Request) -> dict[str, Any]:
        """Process incoming Pub/Sub message for insight generation.

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
                "Processing insight generation for user: %s, upload: %s",
                message_data.get("user_id"),
                message_data.get("upload_id"),
            )

            # Generate insights using Gemini
            insight_request = HealthInsightRequest(
                user_id=message_data["user_id"],
                analysis_results=message_data["analysis_results"],
                context=message_data.get("context"),
            )
            insights = await self.gemini_service.generate_health_insights(
                insight_request
            )

            # Store insights (implementation depends on your storage solution)
            await self._store_insights(
                user_id=message_data["user_id"],
                upload_id=message_data["upload_id"],
                insights=insights.model_dump(),
            )

            self.logger.info(
                "Completed insight generation for user: %s", message_data["user_id"]
            )

            return {
                "status": "success",
                "user_id": message_data["user_id"],
                "upload_id": message_data["upload_id"],
                "insights_generated": True,
            }

        except Exception as e:
            self.logger.exception("Error processing insight request")
            raise HTTPException(
                status_code=500, detail=f"Insight generation failed: {e!s}"
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
            required_fields = ["user_id", "upload_id", "analysis_results"]
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

    async def _store_insights(
        self, user_id: str, upload_id: str, insights: dict[str, Any]
    ) -> None:
        """Store generated insights.

        Args:
            user_id: User identifier
            upload_id: Upload identifier
            insights: Generated insights

        Note:
            This is a placeholder. Implement actual storage logic based on your needs.
        """
        # TODO: Implement actual storage logic
        # Options:
        # - Store in GCS/S3
        # - Store in database
        # - Send to another service
        self.logger.info("Storing insights for user %s, upload %s", user_id, upload_id)

    @staticmethod
    def _raise_invalid_token_error() -> None:
        """Raise HTTPException for invalid token format."""
        raise HTTPException(status_code=401, detail="Invalid token format")

    @staticmethod
    def _raise_missing_field_error(field: str) -> None:
        """Raise ValueError for missing required field."""
        msg = f"Missing required field: {field}"
        raise ValueError(msg)


# Create FastAPI app for insight service
insight_app = FastAPI(title="Health Insight Generation Service")


@insight_app.post("/process-task")
async def process_insight_task(request: Request) -> dict[str, Any]:
    """Handle Pub/Sub push subscription for insight generation."""
    subscriber = get_insight_subscriber()
    return await subscriber.process_insight_request(request)


@insight_app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "insights"}


class InsightSubscriberSingleton:
    """Singleton container for insight subscriber."""

    _instance: InsightSubscriber | None = None

    @classmethod
    def get_instance(cls) -> InsightSubscriber:
        """Get or create insight subscriber instance."""
        if cls._instance is None:
            cls._instance = InsightSubscriber()
        return cls._instance


def get_insight_subscriber() -> InsightSubscriber:
    """Get or create global insight subscriber instance."""
    return InsightSubscriberSingleton.get_instance()
