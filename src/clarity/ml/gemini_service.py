"""Vertex AI Gemini Service for Health Insights Generation.

This service integrates with Google's Vertex AI Gemini 2.5 Pro model
to generate human-like health insights and narratives from ML analysis results.
"""

from datetime import UTC, datetime
import json
import logging
from typing import Any, NoReturn

from pydantic import BaseModel, Field
import vertexai  # type: ignore[import-untyped]
from vertexai.generative_models import (  # type: ignore[import-untyped]
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
)

logger = logging.getLogger(__name__)

# Clinical thresholds as constants
EXCELLENT_SLEEP_EFFICIENCY = 85
GOOD_SLEEP_EFFICIENCY = 75
POOR_SLEEP_EFFICIENCY = 80
CIRCADIAN_THRESHOLD = 0.7
MAX_NARRATIVE_PREVIEW_LENGTH = 300

# Error messages
GEMINI_NOT_INITIALIZED_MSG = "Gemini model not initialized"


class HealthInsightRequest(BaseModel):
    """Request for generating health insights."""

    user_id: str
    analysis_results: dict[str, Any] = Field(description="PAT analysis results")
    context: str | None = Field(None, description="Additional context for insights")
    insight_type: str = Field(
        default="comprehensive", description="Type of insight to generate"
    )


class HealthInsightResponse(BaseModel):
    """Response containing generated health insights."""

    user_id: str
    narrative: str = Field(description="Human-readable health narrative")
    key_insights: list[str] = Field(description="Key insights extracted")
    recommendations: list[str] = Field(description="Actionable recommendations")
    confidence_score: float = Field(description="Confidence in the insights (0-1)")
    generated_at: str = Field(description="Timestamp of generation")


class GeminiService:
    """Service for generating health insights using Vertex AI Gemini."""

    def __init__(
        self, project_id: str | None = None, location: str = "us-central1"
    ) -> None:
        self.project_id = project_id
        self.location = location
        self.client: Any = None
        self.model: GenerativeModel | None = None
        self.is_initialized = False

    @staticmethod
    def _raise_model_not_initialized() -> NoReturn:
        """Raise RuntimeError when model is not initialized."""
        msg = GEMINI_NOT_INITIALIZED_MSG
        raise RuntimeError(msg)

    async def initialize(self) -> None:
        """Initialize the Vertex AI Gemini client."""
        try:
            # Initialize Vertex AI with project and location
            vertexai.init(project=self.project_id, location=self.location)

            # Create Gemini 2.5 Pro model instance
            self.model = GenerativeModel("gemini-2.5-pro")

            self.is_initialized = True
            logger.info("Gemini service initialized successfully")
            logger.info("   • Project ID: %s", self.project_id)
            logger.info("   • Location: %s", self.location)
            logger.info("   • Model: gemini-2.5-pro")

        except Exception:
            logger.exception("Failed to initialize Gemini service")
            raise

    async def generate_health_insights(
        self, request: HealthInsightRequest
    ) -> HealthInsightResponse:
        """Generate health insights from analysis results."""
        if not self.is_initialized:
            await self.initialize()

        # Ensure model is available after initialization
        if self.model is None:
            raise RuntimeError(GEMINI_NOT_INITIALIZED_MSG)

        try:
            # Create health-focused prompt for Gemini
            prompt = self._create_health_insight_prompt(request)

            # Configure generation parameters for health insights
            generation_config = GenerationConfig(
                temperature=0.3,  # Lower temperature for more consistent medical insights
                top_p=0.8,
                top_k=40,
                max_output_tokens=2048,
                response_mime_type="application/json",
            )

            # Configure safety settings for medical content
            safety_settings = [
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                ),
            ]

            # Generate response using Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )

            # Parse the structured response
            return self._parse_gemini_response(response, request.user_id)

        except Exception:
            logger.exception("Failed to generate health insights")
            raise

    @staticmethod
    def _create_health_insight_prompt(request: HealthInsightRequest) -> str:
        """Create a comprehensive prompt for health insight generation."""
        analysis_data = request.analysis_results
        context = request.context or ""

        # Extract key metrics for the prompt
        sleep_efficiency = analysis_data.get("sleep_efficiency", 0)
        circadian_score = analysis_data.get("circadian_rhythm_score", 0)
        depression_risk = analysis_data.get("depression_risk_score", 0)
        total_sleep_time = analysis_data.get("total_sleep_time", 0)
        wake_after_sleep_onset = analysis_data.get("wake_after_sleep_onset", 0)
        sleep_onset_latency = analysis_data.get("sleep_onset_latency", 0)

        return f"""You are a clinical AI assistant specializing in sleep health and wellness analysis.

Analyze the following health data and provide insights in JSON format:

PATIENT DATA:
- Sleep Efficiency: {sleep_efficiency:.1f}%
- Circadian Rhythm Score: {circadian_score:.2f}
- Depression Risk Score: {depression_risk:.2f}
- Total Sleep Time: {total_sleep_time:.1f} hours
- Wake After Sleep Onset: {wake_after_sleep_onset:.1f} minutes
- Sleep Onset Latency: {sleep_onset_latency:.1f} minutes
- Additional Context: {context}

CLINICAL GUIDELINES:
- Sleep Efficiency >85% = Excellent, 75-85% = Good, <75% = Needs attention
- Circadian Score >0.8 = Strong, 0.6-0.8 = Moderate, <0.6 = Irregular
- Depression Risk >0.7 = Elevated, 0.4-0.7 = Moderate, <0.4 = Low

RESPONSE FORMAT (JSON):
{{
    "narrative": "A comprehensive, empathetic health narrative (200-300 words) that explains the analysis in patient-friendly language, highlighting patterns and overall health status.",
    "key_insights": [
        "3-5 specific, actionable insights based on the data",
        "Focus on the most significant findings",
        "Include both positive aspects and areas for improvement"
    ],
    "recommendations": [
        "3-5 evidence-based, actionable recommendations",
        "Prioritize recommendations based on potential impact",
        "Include both immediate and long-term suggestions"
    ],
    "confidence_score": 0.85
}}

Generate insights that are:
1. Medically accurate and evidence-based
2. Patient-friendly and empathetic in tone
3. Actionable and specific
4. Focused on overall wellness and improvement
5. Respectful of the patient's current health status

Respond only with valid JSON."""

    @staticmethod
    def _parse_gemini_response(response: object, user_id: str) -> HealthInsightResponse:
        """Parse and validate Gemini response."""
        try:
            # Extract text from response
            response_text = response.text.strip()  # type: ignore[attr-defined]

            # Parse JSON response
            parsed_response = json.loads(response_text)

            # Validate required fields and provide defaults
            narrative = parsed_response.get(
                "narrative", "Analysis completed successfully."
            )
            key_insights = parsed_response.get("key_insights", [])
            recommendations = parsed_response.get("recommendations", [])
            confidence_score = parsed_response.get("confidence_score", 0.8)

            return HealthInsightResponse(
                user_id=user_id,
                narrative=narrative,
                key_insights=key_insights,
                recommendations=recommendations,
                confidence_score=confidence_score,
                generated_at=datetime.now(UTC).isoformat(),
            )

        except json.JSONDecodeError:
            logger.warning("Failed to parse Gemini JSON response, using fallback")
            # Fallback to extracting insights from text response
            return GeminiService._create_fallback_response(response.text, user_id)  # type: ignore[attr-defined]
        except Exception:
            logger.exception("Error parsing Gemini response")
            raise

    @staticmethod
    def _create_fallback_response(
        response_text: str, user_id: str
    ) -> HealthInsightResponse:
        """Create fallback response when JSON parsing fails."""
        # Simple text parsing fallback
        narrative = (
            response_text[:MAX_NARRATIVE_PREVIEW_LENGTH] + "..."
            if len(response_text) > MAX_NARRATIVE_PREVIEW_LENGTH
            else response_text
        )

        return HealthInsightResponse(
            user_id=user_id,
            narrative=narrative,
            key_insights=[
                "Health analysis completed",
                "Please review detailed results",
            ],
            recommendations=[
                "Consult with healthcare provider",
                "Monitor trends over time",
            ],
            confidence_score=0.7,
            generated_at=datetime.now(UTC).isoformat(),
        )

    @staticmethod
    def _generate_placeholder_narrative(analysis_results: dict[str, Any]) -> str:
        """Generate a placeholder narrative (enhanced fallback method)."""
        sleep_efficiency = analysis_results.get("sleep_efficiency", 0)
        circadian_score = analysis_results.get("circadian_rhythm_score", 0)

        return (
            f"Based on your recent health data analysis, your sleep efficiency is {sleep_efficiency:.1f}% "
            f"and your circadian rhythm regularity score is {circadian_score:.2f}. This suggests a generally "
            f"healthy sleep pattern with room for optimization in your daily routine consistency."
        )

    @staticmethod
    def _extract_key_insights(analysis_results: dict[str, Any]) -> list[str]:
        """Extract key insights from analysis results (enhanced fallback method)."""
        insights: list[str] = []

        sleep_efficiency = analysis_results.get("sleep_efficiency", 0)
        if sleep_efficiency > EXCELLENT_SLEEP_EFFICIENCY:
            insights.append("Excellent sleep quality maintained")
        elif sleep_efficiency > GOOD_SLEEP_EFFICIENCY:
            insights.append("Good sleep quality with minor optimization opportunities")
        else:
            insights.append("Sleep quality needs attention")

        return insights

    @staticmethod
    def _generate_recommendations(analysis_results: dict[str, Any]) -> list[str]:
        """Generate actionable recommendations (enhanced fallback method)."""
        recommendations: list[str] = []

        sleep_efficiency = analysis_results.get("sleep_efficiency", 0)
        if sleep_efficiency < POOR_SLEEP_EFFICIENCY:
            recommendations.extend([
                "Consider establishing a consistent bedtime routine",
                "Limit screen time 1 hour before bed",
            ])

        circadian_score = analysis_results.get("circadian_rhythm_score", 0)
        if circadian_score < CIRCADIAN_THRESHOLD:
            recommendations.extend([
                "Try to maintain consistent sleep and wake times",
                "Get natural sunlight exposure in the morning",
            ])

        return recommendations

    async def health_check(self) -> dict[str, str | bool]:
        """Check the health status of the Gemini service."""
        return {
            "service": "Gemini Service",
            "status": "healthy" if self.is_initialized else "not_initialized",
            "project_id": self.project_id or "not_set",
            "location": self.location,
            "initialized": self.is_initialized,
            "model": "gemini-2.5-pro" if self.model else "not_loaded",
        }
