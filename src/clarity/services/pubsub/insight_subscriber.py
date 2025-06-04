"""Health insight subscriber for processing Pub/Sub messages and generating health insights."""

import base64
import json
import logging
from typing import Any, NoReturn

from fastapi import FastAPI, HTTPException, Request

from clarity.ml.gemini_service import GeminiService, HealthInsightRequest
from clarity.storage.firestore_client import FirestoreClient

# Configure logger
logger = logging.getLogger(__name__)

# Constants for feature analysis requirements
MIN_CARDIO_FEATURES_REQUIRED = 3
MIN_RESPIRATORY_FEATURES_REQUIRED = 4

# Health insight thresholds
HIGH_CONSISTENCY_THRESHOLD = 0.7
MODERATE_CONSISTENCY_THRESHOLD = 0.5

# Create FastAPI app for insight service
insight_app = FastAPI(
    title="Insight Subscriber Service",
    description="Health insight generation via Pub/Sub",
    version="1.0.0",
)


class GeminiInsightGenerator:
    """Generates health insights using Google's Gemini AI model."""

    def __init__(self, project_id: str | None = None) -> None:
        """Initialize the Gemini insight generator."""
        self.logger = logger
        self.firestore_client = FirestoreClient()
        self.gemini_service = GeminiService(project_id=project_id)

        logger.info("✅ Gemini insight generator initialized")

    async def generate_health_insight(
        self, user_id: str, upload_id: str, analysis_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate health insight from analysis results."""
        try:
            logger.info("🧠 Generating health insight for user %s", user_id)

            # Create health insight request
            insight_request = HealthInsightRequest(
                user_id=user_id,
                analysis_results=analysis_results,
                context="Health analysis results",
                insight_type="comprehensive"
            )

            # Generate insight using GeminiService
            try:
                insight_response = await self.gemini_service.generate_health_insights(insight_request)

                # Convert response to dict for consistency
                insight_dict = {
                    "narrative": insight_response.narrative,
                    "key_insights": insight_response.key_insights,
                    "recommendations": insight_response.recommendations,
                    "confidence_score": insight_response.confidence_score,
                    "generated_at": insight_response.generated_at,
                    "user_id": insight_response.user_id
                }
            except Exception as e:
                logger.warning("⚠️ Gemini service failed, using fallback: %s", str(e))
                insight_dict = self._create_fallback_insight(analysis_results)

            # Store the insight in Firestore
            await self._store_insight(user_id, upload_id, insight_dict)

            return {
                "status": "success",
                "message": "Health insight generated successfully",
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("❌ Failed to generate health insight")
            raise HTTPException(
                status_code=500, detail="Failed to generate health insight"
            ) from e

    def _create_fallback_insight(self, analysis_results: dict[str, Any]) -> dict[str, Any]:
        """Create a fallback insight when Gemini is unavailable."""
        try:
            # Extract key metrics for summary
            sleep_efficiency = analysis_results.get("sleep_efficiency", 0)
            total_sleep_time = analysis_results.get("total_sleep_time", 0)
            circadian_score = analysis_results.get("circadian_rhythm_score", 0)

            # Generate basic narrative
            narrative = f"Health analysis completed. Sleep efficiency: {sleep_efficiency:.1f}%, "
            narrative += f"Total sleep time: {total_sleep_time:.1f} hours, "
            narrative += f"Circadian rhythm score: {circadian_score:.2f}."

            # Basic insights based on thresholds
            insights = []
            if sleep_efficiency >= 85:
                insights.append("Excellent sleep efficiency detected")
            elif sleep_efficiency >= 75:
                insights.append("Good sleep quality observed")
            else:
                insights.append("Sleep efficiency could be improved")

            # Basic recommendations
            recommendations = []
            if sleep_efficiency < 75:
                recommendations.append("Consider establishing a consistent bedtime routine")
            if total_sleep_time < 7:
                recommendations.append("Aim for 7-9 hours of sleep per night")

            return {
                "narrative": narrative,
                "key_insights": insights or ["Analysis completed successfully"],
                "recommendations": recommendations or ["Continue monitoring your health patterns"],
                "confidence_score": 0.6,
                "generated_at": "fallback_mode",
                "source": "fallback_algorithm"
            }

        except Exception:
            logger.exception("Failed to create fallback insight")
            return {
                "narrative": "Health analysis completed.",
                "key_insights": ["Analysis results available"],
                "recommendations": ["Continue monitoring your health"],
                "confidence_score": 0.5,
                "generated_at": "fallback_mode",
                "source": "minimal_fallback"
            }

    async def _store_insight(self, user_id: str, upload_id: str, insight: dict[str, Any]) -> None:
        """Store generated insight in Firestore."""
        try:
            document_path = f"users/{user_id}/insights/{upload_id}"
            await self.firestore_client.create_document(
                collection="insights",
                document_id=f"{user_id}_{upload_id}",
                data=insight
            )
            logger.info("✅ Insight stored successfully: %s", document_path)

        except Exception:
            logger.exception("❌ Failed to store insight in Firestore")
            raise


class InsightSubscriber:
    """Handles Pub/Sub messages for health insight generation."""

    def __init__(self, project_id: str | None = None) -> None:
        """Initialize the insight subscriber."""
        self.logger = logger
        self.insight_generator = GeminiInsightGenerator(project_id=project_id)

    async def process_insight_request_message(self, request: Request) -> dict[str, Any]:
        """Process incoming Pub/Sub message for insight generation."""
        try:
            logger.info("📨 Processing insight generation request")

            # Get request body
            pubsub_body = await request.json()
            logger.debug("📨 Received Pub/Sub body: %s", pubsub_body)

            # Extract and validate message data
            message_data = self._extract_message_data(pubsub_body)

            # Generate health insight
            result = await self.insight_generator.generate_health_insight(
                user_id=message_data["user_id"],
                upload_id=message_data["upload_id"],
                analysis_results=message_data["analysis_results"],
            )

            logger.info("✅ Successfully processed insight request")
            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("❌ Failed to process insight request")
            self._raise_processing_error(str(e))

    def _extract_message_data(self, pubsub_body: dict[str, Any]) -> dict[str, Any]:
        """Extract and validate message data from Pub/Sub payload."""
        try:
            # Pub/Sub push format: {"message": {"data": "<base64>", "attributes": {...}}}
            if "message" not in pubsub_body:
                self._raise_missing_field_error("message")

            message = pubsub_body["message"]

            if "data" not in message:
                self._raise_missing_field_error("data")

            # Decode base64 data
            try:
                decoded_data = base64.b64decode(message["data"]).decode("utf-8")
                message_data: dict[str, Any] = json.loads(decoded_data)
            except (base64.binascii.Error, json.JSONDecodeError) as e:
                self._raise_invalid_data_error() from e

            # Validate required fields
            for field in ["user_id", "upload_id", "analysis_results"]:
                if field not in message_data:
                    self._raise_missing_field_error(field)

            return message_data

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("❌ Failed to extract message data")
            self._raise_invalid_format_error() from e

    @staticmethod
    def _raise_missing_field_error(field: str) -> NoReturn:
        """Raise HTTPException for missing required field."""
        detail = f"Missing required field: {field}"
        raise HTTPException(status_code=400, detail=detail)

    @staticmethod
    def _raise_invalid_data_error() -> NoReturn:
        """Raise HTTPException for invalid message data."""
        raise HTTPException(status_code=400, detail="Invalid message data")

    @staticmethod
    def _raise_invalid_format_error() -> NoReturn:
        """Raise HTTPException for invalid message format."""
        raise HTTPException(status_code=400, detail="Invalid message format")

    @staticmethod
    def _raise_processing_error(error_msg: str) -> NoReturn:
        """Raise HTTPException for processing error."""
        raise HTTPException(status_code=500, detail="Failed to process insight request")


# Global subscriber instance
_insight_subscriber: InsightSubscriber | None = None


def get_insight_subscriber() -> InsightSubscriber:
    """Get or create the insight subscriber instance."""
    global _insight_subscriber  # noqa: PLW0603
    if _insight_subscriber is None:
        _insight_subscriber = InsightSubscriber()
    return _insight_subscriber


# FastAPI endpoint handlers
@insight_app.post("/generate-insight")
async def generate_insight_task(request: Request) -> dict[str, Any]:
    """Handle Pub/Sub messages for insight generation."""
    subscriber = get_insight_subscriber()
    return await subscriber.process_insight_request_message(request)


@insight_app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "insight-subscriber"}
