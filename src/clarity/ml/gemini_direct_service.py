"""Direct Google Gemini API service (not through Vertex AI)."""

import asyncio
import logging
from typing import Any

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory

from clarity.core.exceptions import ServiceError

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Google Gemini API directly."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> None:
        """Initialize Gemini service with API key."""
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Configure Gemini with API key
        genai.configure(api_key=api_key)

        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.8,
                "top_k": 10,
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
        )

        logger.info("Initialized Gemini service with model: %s", model_name)

    async def generate_health_insights(
        self, health_data: dict[str, Any], user_context: dict[str, Any] | None = None
    ) -> str:
        """Generate personalized health insights from data."""
        try:
            # Prepare the prompt
            prompt = self._build_health_insights_prompt(health_data, user_context)

            # Generate response
            response = await asyncio.to_thread(self.model.generate_content, prompt)

            if not response.text:
                msg = "Gemini returned empty response"
                raise ServiceError(msg)

            return response.text

        except Exception as e:
            logger.exception("Error generating health insights: %s", e)
            msg = f"Failed to generate insights: {e!s}"
            raise ServiceError(msg)

    async def analyze_pat_results(
        self,
        pat_analysis: dict[str, Any],
        additional_metrics: dict[str, Any] | None = None,
    ) -> str:
        """Generate insights from PAT (Pretrained Actigraphy Transformer) results."""
        try:
            prompt = self._build_pat_analysis_prompt(pat_analysis, additional_metrics)

            response = await asyncio.to_thread(self.model.generate_content, prompt)

            if not response.text:
                msg = "Gemini returned empty response"
                raise ServiceError(msg)

            return response.text

        except Exception as e:
            logger.exception("Error analyzing PAT results: %s", e)
            msg = f"Failed to analyze PAT results: {e!s}"
            raise ServiceError(msg)

    async def generate_recommendations(
        self,
        user_profile: dict[str, Any],
        health_metrics: dict[str, Any],
        goals: list[str] | None = None,
    ) -> list[str]:
        """Generate personalized health recommendations."""
        try:
            prompt = self._build_recommendations_prompt(
                user_profile, health_metrics, goals
            )

            response = await asyncio.to_thread(self.model.generate_content, prompt)

            if not response.text:
                msg = "Gemini returned empty response"
                raise ServiceError(msg)

            # Parse recommendations from response
            return self._parse_recommendations(response.text)

        except Exception as e:
            logger.exception("Error generating recommendations: %s", e)
            msg = f"Failed to generate recommendations: {e!s}"
            raise ServiceError(msg)

    def _build_health_insights_prompt(
        self, health_data: dict[str, Any], user_context: dict[str, Any] | None = None
    ) -> str:
        """Build prompt for health insights generation."""
        prompt = f"""As a health AI assistant, analyze the following health data and provide personalized insights.

Health Data:
{self._format_health_data(health_data)}

"""
        if user_context:
            prompt += f"""User Context:
Age: {user_context.get('age', 'Unknown')}
Gender: {user_context.get('gender', 'Unknown')}
Activity Level: {user_context.get('activity_level', 'Unknown')}
Health Goals: {', '.join(user_context.get('goals', []))}

"""

        prompt += """Provide:
1. Key observations about the health metrics
2. Potential areas of concern (if any)
3. Positive trends or achievements
4. Actionable recommendations

Keep the response concise, supportive, and focused on actionable insights."""

        return prompt

    def _build_pat_analysis_prompt(
        self,
        pat_analysis: dict[str, Any],
        additional_metrics: dict[str, Any] | None = None,
    ) -> str:
        """Build prompt for PAT analysis interpretation."""
        prompt = f"""Interpret the following sleep and activity analysis from the Pretrained Actigraphy Transformer (PAT):

PAT Analysis Results:
- Sleep Quality Score: {pat_analysis.get('sleep_quality_score', 'N/A')}
- Sleep Efficiency: {pat_analysis.get('sleep_efficiency', 'N/A')}%
- Total Sleep Time: {pat_analysis.get('total_sleep_time', 'N/A')} hours
- Sleep Onset Latency: {pat_analysis.get('sleep_onset_latency', 'N/A')} minutes
- Wake After Sleep Onset: {pat_analysis.get('wake_after_sleep_onset', 'N/A')} minutes
- Number of Awakenings: {pat_analysis.get('num_awakenings', 'N/A')}
- Activity Patterns: {pat_analysis.get('activity_pattern', 'N/A')}

"""

        if additional_metrics:
            prompt += f"""Additional Health Metrics:
{self._format_health_data(additional_metrics)}

"""

        prompt += """Provide a comprehensive interpretation that includes:
1. Sleep quality assessment
2. Activity pattern analysis
3. Potential impacts on overall health
4. Specific recommendations for improvement

Use clear, non-technical language suitable for a health-conscious user."""

        return prompt

    def _build_recommendations_prompt(
        self,
        user_profile: dict[str, Any],
        health_metrics: dict[str, Any],
        goals: list[str] | None = None,
    ) -> str:
        """Build prompt for personalized recommendations."""
        prompt = f"""Generate personalized health recommendations based on:

User Profile:
- Age: {user_profile.get('age', 'Unknown')}
- Gender: {user_profile.get('gender', 'Unknown')}
- Activity Level: {user_profile.get('activity_level', 'Moderate')}
- Health Conditions: {', '.join(user_profile.get('conditions', ['None reported']))}

Current Health Metrics:
{self._format_health_data(health_metrics)}

"""

        if goals:
            prompt += f"Health Goals: {', '.join(goals)}\n\n"

        prompt += """Provide 5-7 specific, actionable recommendations that:
1. Are tailored to the user's profile and current health status
2. Support their health goals
3. Are realistic and achievable
4. Include both immediate actions and long-term habits

Format each recommendation as a clear action item with brief explanation."""

        return prompt

    def _format_health_data(self, data: dict[str, Any]) -> str:
        """Format health data for prompt."""
        formatted = []
        for key, value in data.items():
            if isinstance(value, dict):
                formatted.append(f"- {key.replace('_', ' ').title()}:")
                for sub_key, sub_value in value.items():
                    formatted.append(
                        f"  - {sub_key.replace('_', ' ').title()}: {sub_value}"
                    )
            else:
                formatted.append(f"- {key.replace('_', ' ').title()}: {value}")
        return "\n".join(formatted)

    def _parse_recommendations(self, response_text: str) -> list[str]:
        """Parse recommendations from Gemini response."""
        # Simple parsing - split by newlines and filter
        lines = response_text.strip().split("\n")
        recommendations = []

        for line in lines:
            line = line.strip()
            # Look for numbered or bulleted items
            if line and (line[0].isdigit() or line.startswith(("-", "•", "*"))):
                # Clean up the line
                clean_line = line.lstrip("0123456789.-•* ")
                if clean_line:
                    recommendations.append(clean_line)

        return recommendations[:7]  # Return max 7 recommendations
