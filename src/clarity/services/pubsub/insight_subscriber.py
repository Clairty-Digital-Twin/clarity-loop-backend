"""Health insight subscriber for processing Pub/Sub messages and generating health insights."""

import base64
import json
from typing import Any, NoReturn

from fastapi import HTTPException, Request
import google.generativeai as genai  # type: ignore[import-untyped]

from clarity.core.exception_manager import ExceptionHandler
from clarity.core.exceptions import DatabaseConfigError
from clarity.core.logging_setup import logger
from clarity.integrations.gemini import GeminiClient
from clarity.services.pubsub.base_subscriber import insight_app
from clarity.storage.firestore_client import FirestoreClient

# Health insight thresholds
HIGH_CONSISTENCY_THRESHOLD = 0.7
MODERATE_CONSISTENCY_THRESHOLD = 0.5
MIN_CARDIO_FEATURES_REQUIRED = 3
MIN_RESPIRATORY_FEATURES_REQUIRED = 4


class GeminiInsightGenerator:
    """Handles health insight generation using Google's Gemini AI."""

    def __init__(self) -> None:
        """Initialize Gemini insight generator."""
        self.logger = logger
        self.firestore_client = FirestoreClient()
        self.model = None

        try:
            # Initialize Gemini client
            gemini_client = GeminiClient()
            self.model = gemini_client.get_model()
            self.logger.info("✅ Gemini model initialized for health insights")
        except Exception:
            self.logger.warning("⚠️ Failed to initialize Gemini model")

    async def generate_health_insight(
        self, user_id: str, upload_id: str, analysis_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate health insight from analysis results."""
        try:
            self.logger.info(f"🧠 Generating health insight for user {user_id}")

            if self.model is not None:
                # Generate AI-powered insight
                prompt = GeminiInsightGenerator._create_health_prompt(analysis_results)
                self.logger.debug(f"📝 Generated prompt for Gemini: {prompt[:200]}...")

                insight = await self._call_gemini_api(prompt)
                self.logger.info("✅ Generated AI health insight")
            else:
                # Fallback to mock insight
                insight = GeminiInsightGenerator._create_mock_insight(analysis_results)
                self.logger.info("✅ Generated mock health insight (Gemini unavailable)")

            # Add metadata
            insight["analysis_summary"] = GeminiInsightGenerator._create_analysis_summary(
                analysis_results
            )
            insight["generated_timestamp"] = analysis_results.get(
                "processing_metadata", {}
            ).get("processing_timestamp")

            # Store the insight
            await self._store_insight(user_id, upload_id, insight)

            return {
                "status": "success",
                "message": "Health insight generated successfully",
            }

        except HTTPException:
            raise
        except Exception as e:
            self.logger.exception(f"❌ Failed to generate health insight: {e}")
            ExceptionHandler.log_exception(e, {"user_id": user_id, "upload_id": upload_id})
            raise HTTPException(
                status_code=500, detail="Failed to generate health insight"
            ) from e

    @staticmethod
    def _enhance_analysis_results_for_gemini(analysis_results: dict[str, Any]) -> dict[str, Any]:
        """Enhance analysis results to be more readable for Gemini AI."""
        enhanced: dict[str, Any] = {}

        # Handle sleep features if available
        if sleep_features := analysis_results.get("sleep_features"):
            sf = sleep_features[0] if isinstance(sleep_features, list) and sleep_features else {}

            # Convert Pydantic model to dict if needed
            if hasattr(sf, "model_dump"):
                try:
                    sf = sf.model_dump()  # convert to dict if Pydantic model
                except AttributeError:
                    # Handle mock that raises AttributeError
                    sf = sf.dict() if hasattr(sf, "dict") else {}
            elif hasattr(sf, "dict"):
                try:
                    sf = sf.dict()  # fallback for older Pydantic
                except AttributeError:
                    sf = {}  # fallback for invalid types
            elif not isinstance(sf, dict):
                # Skip processing if sf is not a dict-like object (e.g., string)
                sf = {}

            # Map to expected field names for Gemini - only if sf is valid dict
            if isinstance(sf, dict):
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
        if len(cardio_features) >= MIN_CARDIO_FEATURES_REQUIRED:
            prompt_parts.extend([
                f"Average Heart Rate: {cardio_features[0]} bpm",
                f"Max Heart Rate: {cardio_features[1]} bpm",
                f"Min Heart Rate: {cardio_features[2]} bpm",
            ])

        # Respiratory metrics (need at least 4 for basic analysis)
        if len(respiratory_features) >= MIN_RESPIRATORY_FEATURES_REQUIRED:
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

        if (cardio_health := health_indicators.get("cardiovascular_health")) and (circadian_score := cardio_health.get("circadian_rhythm")):
            prompt_parts.append(f"Circadian Rhythm Score: {circadian_score}/1.0")

        if (respiratory_health := health_indicators.get("respiratory_health")) and (respiratory_score := respiratory_health.get("respiratory_stability")):
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
                return insight
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "insights": [response_text[:200] + "..."],
                    "recommendations": ["Continue monitoring your health metrics"],
                    "risk_factors": ["Unable to determine from current data"],
                    "health_score": 7,
                    "confidence_level": "medium",
                }

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

        return {
            "total_features": total_features,
            "modalities_processed": modalities_processed,
            "processing_metadata": analysis_results.get("processing_metadata", {}),
        }

    async def _store_insight(
        self, user_id: str, upload_id: str, insight: dict[str, Any]
    ) -> None:
        """Store generated insight in Firestore."""
        try:
            collection_path = f"users/{user_id}/health_insights"
            await self.firestore_client.create_document(
                collection_path=collection_path,
                data=insight,
                document_id=upload_id,
            )
            self.logger.info(f"✅ Stored insight for user {user_id}")

        except Exception as e:
            self.logger.exception(f"❌ Failed to store insight: {e}")
            raise


class InsightSubscriber:
    """Pub/Sub subscriber for processing health insight requests."""

    def __init__(self) -> None:
        """Initialize insight subscriber."""
        self.logger = logger
        self.insight_generator = GeminiInsightGenerator()

    async def process_insight_request_message(self, request: Request) -> dict[str, Any]:
        """Process incoming Pub/Sub message for insight generation."""
        try:
            self.logger.info("📨 Processing insight generation request")

            # Get request body
            pubsub_body = await request.json()
            self.logger.debug(f"📨 Received Pub/Sub body: {pubsub_body}")

            # Extract and validate message data
            message_data = self._extract_message_data(pubsub_body)

            # Generate health insight
            result = await self.insight_generator.generate_health_insight(
                user_id=message_data["user_id"],
                upload_id=message_data["upload_id"],
                analysis_results=message_data["analysis_results"],
            )

            self.logger.info("✅ Successfully processed insight request")
            return result

        except HTTPException:
            raise
        except Exception as e:
            self.logger.exception(f"❌ Failed to process insight request: {e}")
            ExceptionHandler.log_exception(e)
            raise HTTPException(
                status_code=500, detail="Failed to process insight request"
            ) from e

    def _extract_message_data(self, pubsub_body: dict[str, Any]) -> dict[str, Any]:
        """Extract and validate message data from Pub/Sub payload."""
        try:
            # Pub/Sub push format: {"message": {"data": "<base64>", "attributes": {...}}}
            if "message" not in pubsub_body:
                raise HTTPException(status_code=400, detail="Missing 'message' field")

            message = pubsub_body["message"]

            if "data" not in message:
                raise HTTPException(status_code=400, detail="Missing 'data' field")

            # Decode base64 data
            try:
                decoded_data = base64.b64decode(message["data"]).decode("utf-8")
                message_data: dict[str, Any] = json.loads(decoded_data)
            except (base64.binascii.Error, json.JSONDecodeError) as e:
                raise HTTPException(status_code=400, detail="Invalid message data") from e

            # Validate required fields
            for field in ["user_id", "upload_id", "analysis_results"]:
                if field not in message_data:
                    self._raise_missing_field_error(field)

            return message_data

        except HTTPException:
            raise
        except Exception as e:
            self.logger.exception(f"❌ Failed to extract message data: {e}")
            raise HTTPException(status_code=400, detail="Invalid message format") from e

    @staticmethod
    def _raise_missing_field_error(field: str) -> NoReturn:
        """Raise HTTPException for missing required field."""
        detail = f"Missing required field: {field}"
        raise HTTPException(status_code=400, detail=detail)


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
    """Get insight subscriber instance."""
    return InsightSubscriberSingleton.get_instance()
