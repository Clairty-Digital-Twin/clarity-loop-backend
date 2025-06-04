"""Health insight subscriber for processing Pub/Sub messages and generating health insights."""

import base64
import json
import logging
import os
from typing import Any, NoReturn

from fastapi import FastAPI, HTTPException, Request

from clarity.ml.gemini_service import GeminiService, HealthInsightRequest
from clarity.storage import firestore_client
from clarity.storage.firestore_client import FirestoreClient

# Module level aliases for test compatibility
firestore = firestore_client

# Try to import genai for module level access
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Configure logger
logger = logging.getLogger(__name__)

# Constants for feature analysis requirements
MIN_CARDIO_FEATURES_REQUIRED = 3
MIN_RESPIRATORY_FEATURES_REQUIRED = 4
MIN_FEATURE_VECTOR_LENGTH = 8

# Health insight thresholds
HIGH_CONSISTENCY_THRESHOLD = 0.8
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
        # Use project_id or default for Firestore
        firestore_project_id = project_id or "clarity-digital-twin"
        self.firestore_client = FirestoreClient(project_id=firestore_project_id)
        self.gemini_service = GeminiService(project_id=project_id)

        # For test compatibility - mock model when API key not available
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key and genai is not None:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-pro')
            else:
                self.model = None
        except Exception:
            self.model = None

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
                **insight_dict  # Include the actual insight data
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("❌ Failed to generate health insight")
            raise HTTPException(
                status_code=500, detail="Failed to generate health insight"
            ) from e

    @staticmethod
    def _enhance_analysis_results_for_gemini(analysis_results: dict[str, Any]) -> dict[str, Any]:
        """Enhance analysis results with computed metrics for Gemini."""
        enhanced = analysis_results.copy()

        # Extract sleep features if present
        sleep_features = analysis_results.get("sleep_features")
        if sleep_features is not None:
            # Handle both dict and Pydantic model
            if hasattr(sleep_features, 'model_dump'):
                try:
                    sleep_dict = sleep_features.model_dump()
                except AttributeError:
                    # Fall back to dict() for older Pydantic versions
                    sleep_dict = sleep_features.dict()
            elif hasattr(sleep_features, 'dict'):
                sleep_dict = sleep_features.dict()
            elif isinstance(sleep_features, dict):
                sleep_dict = sleep_features
            else:
                sleep_dict = {}

            # Convert and enhance metrics
            if "sleep_efficiency" in sleep_dict:
                enhanced["sleep_efficiency"] = sleep_dict["sleep_efficiency"] * 100
            if "total_sleep_minutes" in sleep_dict:
                enhanced["total_sleep_time"] = sleep_dict["total_sleep_minutes"] / 60
            if "waso_minutes" in sleep_dict:
                enhanced["wake_after_sleep_onset"] = sleep_dict["waso_minutes"]
            if "sleep_latency" in sleep_dict:
                enhanced["sleep_onset_latency"] = sleep_dict["sleep_latency"]
            if "rem_percentage" in sleep_dict:
                enhanced["rem_sleep_percent"] = sleep_dict["rem_percentage"] * 100
            if "deep_percentage" in sleep_dict:
                enhanced["deep_sleep_percent"] = sleep_dict["deep_percentage"] * 100

            # Sleep consistency rating
            consistency_score = sleep_dict.get("consistency_score", 0)
            if consistency_score > HIGH_CONSISTENCY_THRESHOLD:
                enhanced["sleep_consistency_rating"] = "high"
            elif consistency_score > MODERATE_CONSISTENCY_THRESHOLD:
                enhanced["sleep_consistency_rating"] = "moderate"
            else:
                enhanced["sleep_consistency_rating"] = "low"

        return enhanced

    @staticmethod
    def _create_health_prompt(analysis_results: dict[str, Any]) -> str:
        """Create a health analysis prompt for Gemini."""
        enhanced = GeminiInsightGenerator._enhance_analysis_results_for_gemini(analysis_results)

        prompt_parts = ["Please analyze the following health metrics:"]

        # Add sleep metrics
        if "sleep_efficiency" in enhanced:
            prompt_parts.append(f"Sleep Efficiency: {enhanced['sleep_efficiency']:.1f}%")
        if "total_sleep_time" in enhanced:
            prompt_parts.append(f"Total Sleep Time: {enhanced['total_sleep_time']:.1f} hours")
        if "sleep_consistency_rating" in enhanced:
            prompt_parts.append(f"Sleep Consistency: {enhanced['sleep_consistency_rating'].title()}")

        # Add cardio metrics if sufficient features
        cardio_features = analysis_results.get("cardio_features", [])
        if len(cardio_features) >= MIN_CARDIO_FEATURES_REQUIRED:
            prompt_parts.append(f"Average Heart Rate: {cardio_features[0]:.1f} bpm")
            prompt_parts.append(f"Max Heart Rate: {cardio_features[1]:.1f} bpm")
            prompt_parts.append(f"Min Heart Rate: {cardio_features[2]:.1f} bpm")

        # Add respiratory metrics if sufficient features
        respiratory_features = analysis_results.get("respiratory_features", [])
        if len(respiratory_features) >= MIN_RESPIRATORY_FEATURES_REQUIRED:
            prompt_parts.append(f"Average Respiratory Rate: {respiratory_features[0]:.1f} rpm")
            prompt_parts.append(f"SpO2 Average: {respiratory_features[3]:.1f}%")

        # Add summary stats if available
        summary_stats = analysis_results.get("summary_stats", {})
        health_indicators = summary_stats.get("health_indicators", {})

        cardio_health = health_indicators.get("cardiovascular_health", {})
        if "circadian_rhythm" in cardio_health:
            prompt_parts.append(f"Circadian Rhythm Score: {cardio_health['circadian_rhythm']:.2f}/1.0")

        respiratory_health = health_indicators.get("respiratory_health", {})
        if "respiratory_stability" in respiratory_health:
            prompt_parts.append(f"Respiratory Health Score: {respiratory_health['respiratory_stability']:.2f}/1.0")

        if len(prompt_parts) == 1:  # Only the initial prompt
            prompt_parts.append("No specific health metrics available for analysis.")

        return "\n".join(prompt_parts)

    @staticmethod
    def _raise_model_not_initialized_error() -> NoReturn:
        """Raise RuntimeError for uninitialized model."""
        raise RuntimeError("Gemini model not properly initialized")

    async def _call_gemini_api(self, prompt: str) -> dict[str, Any]:
        """Call Gemini API with the given prompt."""
        try:
            response = self.model.generate_content(prompt)
            # Parse JSON response
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON from Gemini, using fallback")
                return self._create_fallback_insight({})
        except Exception as e:
            logger.exception("Gemini API call failed")
            raise e

    @staticmethod
    def _create_mock_insight(analysis_results: dict[str, Any]) -> dict[str, Any]:
        """Create a mock insight for testing or when Gemini is unavailable."""
        return {
            "insights": [
                "Mock insight generated for testing",
                "Health analysis completed successfully"
            ],
            "recommendations": [
                "Continue monitoring your health metrics",
                "Maintain consistent sleep schedule"
            ],
            "risk_factors": ["None identified"],
            "health_score": 75,
            "confidence_level": "medium"
        }

    @staticmethod
    def _create_analysis_summary(analysis_results: dict[str, Any]) -> dict[str, Any]:
        """Create a summary of the analysis results."""
        total_features = 0
        modalities = []

        for key, value in analysis_results.items():
            if key.endswith("_features") and isinstance(value, list):
                total_features += len(value)
                modality_name = key.replace("_features", "")
                modalities.append(modality_name)

        return {
            "total_features": total_features,
            "modalities_processed": modalities,
            "processing_metadata": {
                "total_data_points": len(analysis_results),
                "has_sleep_data": "sleep_features" in analysis_results
            }
        }

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
        self.generator = GeminiInsightGenerator(project_id=project_id)

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
            result = await self.generator.generate_health_insight(
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
                self._raise_missing_field_http_error("message")

            message = pubsub_body["message"]

            if "data" not in message:
                self._raise_missing_field_http_error("data")

            # Decode base64 data
            try:
                decoded_data = base64.b64decode(message["data"]).decode("utf-8")
                message_data: dict[str, Any] = json.loads(decoded_data)
            except (ValueError, json.JSONDecodeError):
                self._raise_invalid_data_error()

            # Validate required fields
            for field in ["user_id", "upload_id", "analysis_results"]:
                if field not in message_data:
                    self._raise_missing_field_http_error(field)

            return message_data

        except HTTPException:
            raise
        except Exception:
            logger.exception("❌ Failed to extract message data")
            self._raise_invalid_format_error()

    @staticmethod
    def _raise_missing_field_error(field: str) -> NoReturn:
        """Raise ValueError for missing required field (test compatibility)."""
        raise ValueError(f"Missing required field: {field}")

    @staticmethod
    def _raise_missing_field_http_error(field: str) -> NoReturn:
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


class InsightSubscriberSingleton:
    """Singleton pattern for InsightSubscriber."""

    _instance: InsightSubscriber | None = None

    @classmethod
    def get_instance(cls) -> InsightSubscriber:
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = InsightSubscriber()
        return cls._instance


# Global subscriber instance
_insight_subscriber: InsightSubscriber | None = None


def get_insight_subscriber() -> InsightSubscriber:
    """Get or create the insight subscriber instance."""
    return InsightSubscriberSingleton.get_instance()


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
