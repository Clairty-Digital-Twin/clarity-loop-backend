"""Insight Subscriber Service.

Handles Pub/Sub messages for AI-powered health insight generation using Gemini 2.5.
"""

import base64
from datetime import UTC, datetime
import json
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from google.cloud import firestore
import google.generativeai as genai

logger = logging.getLogger(__name__)


class GeminiInsightGenerator:
    """AI-powered insight generator using Gemini 2.5."""

    def __init__(self) -> None:
        """Initialize Gemini insight generator."""
        self.logger = logging.getLogger(__name__)

        # Initialize Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-1.5-pro")
        else:
            self.logger.warning("GEMINI_API_KEY not found - using mock responses")
            self.model = None

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

            # Create prompt from analysis results
            prompt = self._create_health_prompt(analysis_results)

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

            self.logger.info(
                "Generated and stored health insight for user: %s", user_id
            )

            return insight

        except Exception as e:
            self.logger.exception("Failed to generate health insight")
            raise

    @staticmethod
    def _create_health_prompt(analysis_results: dict[str, Any]) -> str:
        """Create structured prompt for Gemini from analysis results."""
        # Extract key metrics
        metrics_lines = []

        # Cardiovascular metrics
        if analysis_results.get("cardio_features"):
            cardio = analysis_results["cardio_features"]
            if len(cardio) >= 8:
                metrics_lines.extend(
                    [
                        f"- Average Heart Rate: {cardio[0]:.1f} bpm",
                        f"- Resting Heart Rate: {cardio[2]:.1f} bpm",
                        f"- Maximum Heart Rate: {cardio[1]:.1f} bpm",
                        f"- Heart Rate Variability: {cardio[4]:.1f} ms",
                        f"- Heart Rate Recovery Score: {cardio[6]:.2f}/1.0",
                        f"- Circadian Rhythm Score: {cardio[7]:.2f}/1.0",
                    ]
                )

        # Respiratory metrics
        if analysis_results.get("respiratory_features"):
            resp = analysis_results["respiratory_features"]
            if len(resp) >= 8:
                metrics_lines.extend(
                    [
                        f"- Average Respiratory Rate: {resp[0]:.1f} breaths/min",
                        f"- Resting Respiratory Rate: {resp[1]:.1f} breaths/min",
                        f"- Average Oxygen Saturation: {resp[3]:.1f}%",
                        f"- Minimum Oxygen Saturation: {resp[4]:.1f}%",
                        f"- Respiratory Stability Score: {resp[6]:.2f}/1.0",
                        f"- Oxygenation Efficiency Score: {resp[7]:.2f}/1.0",
                    ]
                )

        # Activity metrics (from PAT analysis)
        if "summary_stats" in analysis_results:
            summary = analysis_results["summary_stats"]
            if "health_indicators" in summary:
                indicators = summary["health_indicators"]

                if "cardiovascular_health" in indicators:
                    cv = indicators["cardiovascular_health"]
                    metrics_lines.append(
                        f"- Cardiovascular Health Score: {cv.get('circadian_rhythm', 0):.2f}/1.0"
                    )

                if "respiratory_health" in indicators:
                    resp_health = indicators["respiratory_health"]
                    metrics_lines.append(
                        f"- Respiratory Health Score: {resp_health.get('respiratory_stability', 0):.2f}/1.0"
                    )

        metrics_summary = (
            "\\n".join(metrics_lines)
            if metrics_lines
            else "Limited health data available"
        )

        # Create structured prompt
        return f"""You are a health AI assistant analyzing a user's health data. Generate a comprehensive health insight report.

HEALTH DATA SUMMARY:
{metrics_summary}

INSTRUCTIONS:
1. Analyze the health metrics and identify key patterns
2. Provide actionable health recommendations
3. Maintain an encouraging and supportive tone
4. Focus on trends and improvements, not just current values
5. Respond in JSON format with the following structure:

{{
    "narrative": "A comprehensive 2-3 paragraph summary of the user's health status",
    "key_insights": [
        "Specific insight about cardiovascular health",
        "Specific insight about respiratory health",
        "Specific insight about activity patterns",
        "Overall health trend observation"
    ],
    "recommendations": [
        "Specific actionable recommendation #1",
        "Specific actionable recommendation #2",
        "Specific actionable recommendation #3"
    ],
    "health_score": "Overall health score from 1-10",
    "confidence_level": "Confidence in analysis (high/medium/low)"
}}

Generate the health insight now:"""

    async def _call_gemini_api(self, prompt: str) -> dict[str, Any]:
        """Call Gemini API to generate insight."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
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
                insight = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                insight = {
                    "narrative": response_text,
                    "key_insights": [response_text[:200] + "..."],
                    "recommendations": ["Continue monitoring your health metrics"],
                    "health_score": "7",
                    "confidence_level": "medium",
                }

            return insight

        except Exception as e:
            self.logger.exception("Gemini API call failed")
            raise

    @staticmethod
    def _create_mock_insight(analysis_results: dict[str, Any]) -> dict[str, Any]:
        """Create mock insight when Gemini is not available."""
        return {
            "narrative": "Your health data shows generally positive trends. Your cardiovascular and respiratory metrics are within healthy ranges, indicating good overall fitness. Continue maintaining your current activity levels and healthy lifestyle habits.",
            "key_insights": [
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
            "health_score": "8",
            "confidence_level": "medium",
        }

    def _create_analysis_summary(
        self, analysis_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Create summary of analysis results for storage."""
        summary = {
            "modalities_processed": [],
            "feature_counts": {},
            "processing_metadata": analysis_results.get("processing_metadata", {}),
        }

        if analysis_results.get("cardio_features"):
            summary["modalities_processed"].append("cardiovascular")
            summary["feature_counts"]["cardio"] = len(
                analysis_results["cardio_features"]
            )

        if analysis_results.get("respiratory_features"):
            summary["modalities_processed"].append("respiratory")
            summary["feature_counts"]["respiratory"] = len(
                analysis_results["respiratory_features"]
            )

        if analysis_results.get("activity_embedding"):
            summary["modalities_processed"].append("activity")
            summary["feature_counts"]["activity"] = len(
                analysis_results["activity_embedding"]
            )

        if analysis_results.get("fused_vector"):
            summary["feature_counts"]["fused"] = len(analysis_results["fused_vector"])

        return summary

    async def _store_insight(
        self, user_id: str, upload_id: str, insight: dict[str, Any]
    ) -> None:
        """Store insight in Firestore."""
        try:
            # Store in insights/{user_id}/uploads/{upload_id}
            doc_ref = (
                self.firestore_client.collection("insights")
                .document(user_id)
                .collection("uploads")
                .document(upload_id)
            )

            doc_ref.set(insight)

            self.logger.info(
                "Stored insight in Firestore: user=%s, upload=%s", user_id, upload_id
            )

        except Exception as e:
            self.logger.exception("Failed to store insight in Firestore")
            raise


class InsightSubscriber:
    """Subscriber service for insight generation."""

    def __init__(self) -> None:
        """Initialize insight subscriber."""
        self.logger = logging.getLogger(__name__)
        self.insight_generator = GeminiInsightGenerator()

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
            insight = await self.insight_generator.generate_health_insight(
                user_id=message_data["user_id"],
                upload_id=message_data["upload_id"],
                analysis_results=message_data["analysis_results"],
            )

            self.logger.info(
                "Completed insight generation for user: %s", message_data["user_id"]
            )

            return {
                "status": "success",
                "user_id": message_data["user_id"],
                "upload_id": message_data["upload_id"],
                "insight_generated": True,
                "insight_id": f"{message_data['user_id']}/{message_data['upload_id']}",
            }

        except Exception as e:
            self.logger.exception("Error processing insight request message: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Insight generation failed: {e!s}"
            )

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
                    msg = f"Missing required field: {field}"
                    raise ValueError(msg)

            return message_data

        except Exception as e:
            self.logger.exception("Failed to extract message data: %s", e)
            raise HTTPException(
                status_code=400, detail=f"Invalid message format: {e!s}"
            )


# Create FastAPI app for insight service
insight_app = FastAPI(title="Health Insight Generation Service")
insight_subscriber = InsightSubscriber()


@insight_app.post("/generate-insight")
async def generate_insight_task(request: Request) -> dict[str, Any]:
    """Handle Pub/Sub push subscription for insight generation."""
    return await insight_subscriber.process_insight_request_message(request)


@insight_app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "insight"}


# Global subscriber instance
_insight_subscriber: InsightSubscriber | None = None


def get_insight_subscriber() -> InsightSubscriber:
    """Get or create global insight subscriber instance."""
    global _insight_subscriber

    if _insight_subscriber is None:
        _insight_subscriber = InsightSubscriber()

    return _insight_subscriber
