"""Insight Subscriber Service.

Handles Pub/Sub messages for AI-powered health insight generation using Gemini 2.5.
"""

import base64
from datetime import UTC, datetime
import json
import logging
import os
from typing import Any, NoReturn

from fastapi import FastAPI, HTTPException, Request
from google.cloud import firestore  # type: ignore[attr-defined]
import google.generativeai as genai

# Constants
MIN_FEATURE_VECTOR_LENGTH = 8
HIGH_CONSISTENCY_THRESHOLD = 0.8
MODERATE_CONSISTENCY_THRESHOLD = 0.5

logger = logging.getLogger(__name__)


class GeminiInsightGenerator:
    """AI-powered insight generator using Gemini 2.5."""

    def __init__(self) -> None:
        """Initialize Gemini insight generator."""
        self.logger = logging.getLogger(__name__)

        # Initialize Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)  # type: ignore[attr-defined]
            self.model = genai.GenerativeModel("gemini-1.5-pro")  # type: ignore[attr-defined]
        else:
            self.logger.warning("GEMINI_API_KEY not found - using mock responses")
            self.model = None  # type: ignore[assignment]

        # Initialize Firestore
        self.firestore_client = firestore.Client()

        self.logger.info("Initialized Gemini insight generator")

    async def generate_health_insight(
        self, user_id: str, upload_id: str, analysis_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate health insight from analysis results.

        Args:
            user_id: User identifier
            upload_id: Upload identifier
            analysis_results: Results from analysis pipeline

        Returns:
            Generated insight
        """
        try:
            self.logger.info("Generating health insight for user: %s", user_id)

            # 🚀 FIXED: Map sleep_features to expected Gemini field names
            enhanced_results = self._enhance_analysis_results_for_gemini(analysis_results)

            # Create prompt from enhanced analysis results
            prompt = self._create_health_prompt(enhanced_results)

            # Generate insight using Gemini
            if self.model:
                insight = await self._call_gemini_api(prompt)
            else:
                insight = self._create_mock_insight(analysis_results)

            # Add metadata
            insight.update(
                {
                    "user_id": user_id,
                    "upload_id": upload_id,
                    "generated_at": datetime.now(UTC).isoformat(),
                    "model": "gemini-1.5-pro" if self.model else "mock",
                    "analysis_summary": self._create_analysis_summary(analysis_results),
                }
            )

            # Store insight in Firestore
            await self._store_insight(user_id, upload_id, insight)

        except Exception:
            self.logger.exception("Failed to generate health insight")
            raise
        else:
            self.logger.info(
                "Generated and stored health insight for user: %s", user_id
            )

            return insight  # type: ignore[no-any-return]

    @staticmethod
    def _enhance_analysis_results_for_gemini(analysis_results: dict[str, Any]) -> dict[str, Any]:
        """Enhance analysis results with fields expected by Gemini service.

        Maps sleep_features to the specific field names that Gemini expects.
        """
        enhanced = analysis_results.copy()

        # 🚀 FIXED: Map sleep_features to expected Gemini field names
        if "sleep_features" in analysis_results:
            sf = analysis_results["sleep_features"]
            # Ensure BaseModel is dict
            if hasattr(sf, "model_dump"):
                sf = sf.model_dump()  # convert to dict if Pydantic model
            elif hasattr(sf, "dict"):
                sf = sf.dict()  # fallback for older Pydantic

            # Map to expected field names for Gemini
            enhanced["sleep_efficiency"] = sf.get("sleep_efficiency", 0) * 100  # as percentage
            enhanced["total_sleep_time"] = (sf.get("total_sleep_minutes", 0) / 60)  # in hours
            enhanced["wake_after_sleep_onset"] = sf.get("waso_minutes", 0)
            enhanced["sleep_onset_latency"] = sf.get("sleep_latency", 0)
            enhanced["rem_sleep_percent"] = sf.get("rem_percentage", 0) * 100
            enhanced["deep_sleep_percent"] = sf.get("deep_percentage", 0) * 100

            # Provide consistency in a user-friendly way
            cons_score = sf.get("consistency_score", 0)
            enhanced["sleep_consistency_rating"] = (
                "high" if cons_score > HIGH_CONSISTENCY_THRESHOLD
                else "moderate" if cons_score > MODERATE_CONSISTENCY_THRESHOLD
                else "low"
            )

        return enhanced

    @staticmethod
    def _raise_model_not_initialized_error() -> NoReturn:
        """Raise error when Gemini model is not initialized."""
        msg = "Gemini model not properly initialized"
        raise RuntimeError(msg)

    @staticmethod
    def _create_health_prompt(analysis_results: dict[str, Any]) -> str:
        """Create health analysis prompt for Gemini."""
        # First, enhance the results for better prompt generation
        enhanced = GeminiInsightGenerator._enhance_analysis_results_for_gemini(
            analysis_results
        )

        cardio_features = analysis_results.get("cardio_features", [])
        respiratory_features = analysis_results.get("respiratory_features", [])

        # Build prompt with available metrics
        prompt_parts = ["Analyze the following health metrics and provide insights:\n"]

        # Cardiovascular metrics (need at least 3 for basic analysis)
        if len(cardio_features) >= 3:
            prompt_parts.extend([
                f"Average Heart Rate: {cardio_features[0]} bpm",
                f"Max Heart Rate: {cardio_features[1]} bpm",
                f"Min Heart Rate: {cardio_features[2]} bpm",
            ])

        # Respiratory metrics (need at least 4 for basic analysis)
        if len(respiratory_features) >= 4:
            prompt_parts.extend([
                f"Average Respiratory Rate: {respiratory_features[0]} rpm",
                f"SpO2 Average: {respiratory_features[3]}%",
            ])

        # Enhanced sleep metrics if available
        if "sleep_efficiency" in enhanced:
            prompt_parts.extend([
                f"Sleep Efficiency: {enhanced['sleep_efficiency']}%",
                f"Total Sleep Time: {enhanced['total_sleep_time']} hours",
            ])

        if "sleep_consistency_rating" in enhanced:
            prompt_parts.append(f"Sleep Consistency: {enhanced['sleep_consistency_rating'].title()}")

        # Health scores from summary stats
        summary_stats = analysis_results.get("summary_stats", {})
        health_indicators = summary_stats.get("health_indicators", {})
        
        if cardio_health := health_indicators.get("cardiovascular_health"):
            if circadian_score := cardio_health.get("circadian_rhythm"):
                prompt_parts.append(f"Circadian Rhythm Score: {circadian_score}/1.0")

        if respiratory_health := health_indicators.get("respiratory_health"):
            if respiratory_score := respiratory_health.get("respiratory_stability"):
                prompt_parts.append(f"Respiratory Health Score: {respiratory_score}/1.0")

        # If no meaningful metrics found, provide general prompt
        if len(prompt_parts) == 1:  # Only has the initial prompt part
            prompt_parts.append("Limited health metrics available for analysis.")

        prompt_parts.append("""

Please provide a comprehensive health analysis with:
1. Key insights about health patterns and trends
2. Specific recommendations for improvement
3. Risk factors or areas of concern (if any)  
4. Overall health score (1-10)

Respond in JSON format with these fields:
{
  "insights": ["insight1", "insight2", ...],
  "recommendations": ["rec1", "rec2", ...], 
  "risk_factors": ["risk1", "risk2", ...],
  "health_score": 8,
  "confidence_level": "high"
}""")

        return "\n".join(prompt_parts)

    async def _call_gemini_api(self, prompt: str) -> dict[str, Any]:
        """Call Gemini API to generate insight."""
        if self.model is None:
            GeminiInsightGenerator._raise_model_not_initialized_error()

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(  # type: ignore[attr-defined]
                    temperature=0.3, max_output_tokens=1024, top_p=0.8
                ),
            )

            # Parse JSON response
            response_text = response.text.strip()

            # Try to extract JSON from response
            if response_text.startswith("```json"):
                response_text = (
                    response_text.replace("```json", "").replace("```", "").strip()
                )

            try:
                insight: dict[str, Any] = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                insight = {
                    "insights": [response_text[:200] + "..."],
                    "recommendations": ["Continue monitoring your health metrics"],
                    "risk_factors": ["Unable to determine from current data"],
                    "health_score": 7,
                    "confidence_level": "medium",
                }

            return insight

        except Exception:
            self.logger.exception("Gemini API call failed")
            raise

    @staticmethod
    def _create_mock_insight(_analysis_results: dict[str, Any]) -> dict[str, Any]:
        """Create mock insight when Gemini is not available."""
        return {
            "insights": [
                "Your heart rate patterns show good cardiovascular fitness",
                "Respiratory efficiency is within optimal ranges",
                "Activity levels demonstrate consistent healthy habits",
                "Overall health trends are positive and stable",
            ],
            "recommendations": [
                "Maintain your current exercise routine and activity levels",
                "Continue monitoring your heart rate during activities",
                "Ensure adequate sleep for optimal recovery",
            ],
            "risk_factors": [
                "No significant risk factors identified from current data",
            ],
            "health_score": 8,
            "confidence_level": "medium",
        }

    @staticmethod
    def _create_analysis_summary(analysis_results: dict[str, Any]) -> dict[str, Any]:
        """Create summary of analysis results for storage."""
        total_features = 0
        modalities_processed = []

        # Count features from different modalities
        for key, value in analysis_results.items():
            if isinstance(value, list) and key.endswith("_features"):
                total_features += len(value)
                modality_name = key.replace("_features", "")
                modalities_processed.append(modality_name)

        summary = {
            "total_features": total_features,
            "modalities_processed": modalities_processed,
            "processing_metadata": analysis_results.get("processing_metadata", {}),
        }

        return summary

    async def _store_insight(
        self, user_id: str, upload_id: str, insight: dict[str, Any]
    ) -> None:
        """Store insight in Firestore."""
        try:
            doc_ref = (
                self.firestore_client.collection("users")
                .document(user_id)
                .collection("insights")
                .document(upload_id)
            )

            doc_ref.set(insight)

            self.logger.info(
                "Stored insight in Firestore: user=%s, upload=%s", user_id, upload_id
            )

        except Exception:
            self.logger.exception("Failed to store insight in Firestore")
            raise


class InsightSubscriber:
    """Subscriber service for insight generation."""

    def __init__(self) -> None:
        """Initialize insight subscriber."""
        self.logger = logging.getLogger(__name__)
        self.generator = GeminiInsightGenerator()

        # Environment settings
        self.environment = os.getenv("ENVIRONMENT", "development")

        self.logger.info("Initialized insight subscriber (env: %s)", self.environment)

    async def process_insight_request_message(self, request: Request) -> dict[str, Any]:
        """Process incoming Pub/Sub message for insight generation.

        Args:
            request: FastAPI request object containing Pub/Sub message

        Returns:
            Processing result
        """
        try:
            # Parse Pub/Sub message
            body = await request.json()
            message_data = self._extract_message_data(body)

            self.logger.info(
                "Processing insight request for user: %s, upload: %s",
                message_data.get("user_id"),
                message_data.get("upload_id"),
            )

            # Generate insight
            await self.generator.generate_health_insight(
                user_id=message_data["user_id"],
                upload_id=message_data["upload_id"],
                analysis_results=message_data["analysis_results"],
            )

            self.logger.info(
                "Completed insight generation for user: %s", message_data["user_id"]
            )

            return {
                "status": "success",
                "message": "Health insight generated successfully",
            }

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            self.logger.exception("Error processing insight request message")
            raise HTTPException(
                status_code=500, detail=f"Insight generation failed: {e!s}"
            ) from None

    def _extract_message_data(self, pubsub_body: dict[str, Any]) -> dict[str, Any]:
        """Extract and decode Pub/Sub message data."""
        try:
            # Pub/Sub push format: {"message": {"data": "<base64>", "attributes": {...}}}
            if "message" not in pubsub_body:
                raise HTTPException(status_code=400, detail="Missing 'message' field")

            message = pubsub_body["message"]

            if "data" not in message:
                raise HTTPException(status_code=400, detail="Missing 'data' field")

            # Decode base64 data
            encoded_data = message["data"]
            decoded_data = base64.b64decode(encoded_data).decode("utf-8")

            # Parse JSON
            message_data: dict[str, Any] = json.loads(decoded_data)

            # Validate required fields
            required_fields = ["user_id", "upload_id", "analysis_results"]
            for field in required_fields:
                if field not in message_data:
                    self._raise_missing_field_error(field)

            return message_data

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            self.logger.exception("Failed to extract message data")
            raise HTTPException(
                status_code=500, detail=f"Invalid message format: {e!s}"
            ) from None

    @staticmethod
    def _raise_missing_field_error(field: str) -> NoReturn:
        """Raise error for missing required field."""
        msg = f"Missing required field: {field}"
        raise ValueError(msg)


# Create FastAPI app for insight service
insight_app = FastAPI(title="Health Insight Generation Service")


@insight_app.post("/generate-insight")
async def generate_insight_task(request: Request) -> dict[str, Any]:
    """Handle Pub/Sub push subscription for insight generation."""
    subscriber = get_insight_subscriber()
    return await subscriber.process_insight_request_message(request)


@insight_app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "insight"}


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
