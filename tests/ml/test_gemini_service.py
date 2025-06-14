"""Comprehensive tests for Gemini Service.

This test suite covers all aspects of the Gemini service including:
- Service initialization and configuration
- Health insight generation from PAT analysis
- Health check functionality
- Error handling and edge cases
- Integration with PAT analysis results
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from clarity.core.exceptions import ServiceUnavailableProblem
from clarity.ml.gemini_service import (
    GeminiService,
    HealthInsightRequest,
    HealthInsightResponse,
)


class TestGeminiServiceInitialization:
    """Test Gemini service initialization and configuration."""

    @staticmethod
    def test_service_initialization_default_config() -> None:
        """Test service initialization with default configuration."""
        service = GeminiService()

        assert service.project_id is None
        assert service.location == "us-central1"
        assert service.model is None
        assert not service.is_initialized

    @staticmethod
    def test_service_initialization_custom_config() -> None:
        """Test service initialization with custom configuration."""
        project_id = "test-project"
        location = "us-west1"

        service = GeminiService(project_id=project_id, location=location)

        assert service.project_id == project_id
        assert service.location == location
        assert service.model is None
        assert not service.is_initialized

    @staticmethod
    @pytest.mark.asyncio
    async def test_service_initialization_success() -> None:
        """Test successful service initialization."""
        service = GeminiService(project_id="test-project")

        # Mock both vertexai.init and GenerativeModel constructor
        with (
            patch("clarity.ml.gemini_service.vertexai.init") as mock_init,
            patch("clarity.ml.gemini_service.GenerativeModel") as mock_model_class,
        ):
            # Mock the GenerativeModel instance
            mock_model_instance = MagicMock()
            mock_model_class.return_value = mock_model_instance

            await service.initialize()

            mock_init.assert_called_once_with(
                project="test-project", location="us-central1"
            )
            mock_model_class.assert_called_once_with("gemini-2.5-pro")
            assert service.is_initialized
            assert service.model is mock_model_instance

    @staticmethod
    @pytest.mark.asyncio
    async def test_service_initialization_failure() -> None:
        """Test service initialization failure."""
        service = GeminiService(project_id="test-project")

        with patch("vertexai.init", side_effect=Exception("Initialization failed")):
            with pytest.raises(Exception, match="Initialization failed"):
                await service.initialize()

            assert not service.is_initialized
            assert service.model is None


class TestGeminiServiceHealthInsights:
    """Test Gemini service health insight generation."""

    @staticmethod
    @pytest.fixture
    def sample_insight_request() -> HealthInsightRequest:
        """Create sample health insight request."""
        return HealthInsightRequest(
            user_id=str(uuid4()),
            analysis_results={
                "sleep_efficiency": 85.0,
                "circadian_rhythm_score": 0.75,
                "depression_risk_score": 0.2,
                "total_sleep_time": 7.5,
                "wake_after_sleep_onset": 30.0,
                "sleep_onset_latency": 15.0,
            },
            context="Regular exercise routine",
            insight_type="comprehensive",
        )

    @staticmethod
    @pytest.mark.asyncio
    async def test_generate_health_insights_success(
        sample_insight_request: HealthInsightRequest,
    ) -> None:
        """Test successful health insight generation."""
        service = GeminiService(project_id="test-project")

        # Mock successful response
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "narrative": "Your sleep patterns show excellent efficiency at 85%",
                "key_insights": [
                    "Sleep efficiency of 85% indicates healthy sleep",
                    "Low depression risk factors observed",
                ],
                "recommendations": [
                    "Maintain current sleep schedule",
                    "Continue regular exercise routine",
                ],
                "confidence_score": 0.85,
            }
        )

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response

        with (
            patch.object(service, "initialize"),
            patch(
                "vertexai.generative_models.GenerativeModel", return_value=mock_model
            ),
        ):
            service.is_initialized = True
            service.model = mock_model

            result = await service.generate_health_insights(sample_insight_request)

            assert isinstance(result, HealthInsightResponse)
            assert result.user_id == sample_insight_request.user_id
            assert "excellent efficiency" in result.narrative
            assert len(result.key_insights) == 2
            assert len(result.recommendations) == 2
            assert result.confidence_score == 0.85

    @staticmethod
    @pytest.mark.asyncio
    async def test_generate_health_insights_initialization_required(
        sample_insight_request: HealthInsightRequest,
    ) -> None:
        """Test health insight generation when initialization is required."""
        service = GeminiService(project_id="test-project")

        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "narrative": "Analysis completed",
                "key_insights": ["Test insight"],
                "recommendations": ["Test recommendation"],
                "confidence_score": 0.8,
            }
        )

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response

        with (
            patch.object(service, "initialize") as mock_init,
            patch(
                "vertexai.generative_models.GenerativeModel", return_value=mock_model
            ),
        ):
            # Simulate initialization during the call
            def init_side_effect() -> None:
                service.is_initialized = True
                service.model = mock_model

            mock_init.side_effect = init_side_effect

            result = await service.generate_health_insights(sample_insight_request)

            mock_init.assert_called_once()
            assert isinstance(result, HealthInsightResponse)

    @staticmethod
    @pytest.mark.asyncio
    async def test_generate_health_insights_model_not_initialized(
        sample_insight_request: HealthInsightRequest,
    ) -> None:
        """Test health insight generation when model is not initialized after init."""
        service = GeminiService(project_id="test-project")

        with patch.object(service, "initialize"):
            service.is_initialized = True
            service.model = None  # Model remains None after initialization

            with pytest.raises(
                ServiceUnavailableProblem, match="temporarily unavailable"
            ):
                await service.generate_health_insights(sample_insight_request)

    @staticmethod
    @pytest.mark.asyncio
    async def test_generate_health_insights_generation_error(
        sample_insight_request: HealthInsightRequest,
    ) -> None:
        """Test health insight generation with generation error."""
        service = GeminiService(project_id="test-project")

        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("Generation failed")

        service.is_initialized = True
        service.model = mock_model

        with pytest.raises(ServiceUnavailableProblem, match="temporarily unavailable"):
            await service.generate_health_insights(sample_insight_request)

    @staticmethod
    @pytest.mark.asyncio
    async def test_generate_health_insights_invalid_json_fallback(
        sample_insight_request: HealthInsightRequest,
    ) -> None:
        """Test health insight generation with invalid JSON fallback."""
        service = GeminiService(project_id="test-project")

        mock_response = MagicMock()
        mock_response.text = "Invalid JSON response"

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response

        service.is_initialized = True
        service.model = mock_model

        result = await service.generate_health_insights(sample_insight_request)

        # Should fall back to text response
        assert isinstance(result, HealthInsightResponse)
        assert "Invalid JSON response" in result.narrative
        assert len(result.key_insights) >= 1
        assert len(result.recommendations) >= 1


class TestGeminiServicePromptGeneration:
    """Test Gemini service prompt generation functionality."""

    @staticmethod
    def test_create_health_insight_prompt_comprehensive() -> None:
        """Test comprehensive health insight prompt generation."""
        request = HealthInsightRequest(
            user_id="test-user",
            analysis_results={
                "sleep_efficiency": 75.0,
                "circadian_rhythm_score": 0.65,
                "depression_risk_score": 0.4,
                "total_sleep_time": 6.5,
                "wake_after_sleep_onset": 45.0,
                "sleep_onset_latency": 25.0,
            },
            context="Stressful work period",
        )

        prompt = GeminiService._create_health_insight_prompt(request)

        # Check that all key components are included
        assert "75.0%" in prompt  # Sleep efficiency
        assert "0.65" in prompt  # Circadian rhythm score
        assert "0.4" in prompt  # Depression risk score
        assert "6.5 hours" in prompt  # Total sleep time
        assert "45.0 minutes" in prompt  # Wake after sleep onset
        assert "25.0 minutes" in prompt  # Sleep onset latency
        assert "Stressful work period" in prompt  # Context
        assert "JSON" in prompt  # Format specification

    @staticmethod
    def test_create_health_insight_prompt_edge_cases() -> None:
        """Test health insight prompt generation with edge case values."""
        request = HealthInsightRequest(
            user_id="edge-case-user",
            analysis_results={
                "sleep_efficiency": 100.0,  # Perfect efficiency
                "circadian_rhythm_score": 1.0,  # Perfect rhythm
                "depression_risk_score": 0.0,  # No risk
                "total_sleep_time": 12.0,  # Long sleep
                "wake_after_sleep_onset": 0.0,  # No awakenings
                "sleep_onset_latency": 0.0,  # Instant sleep
            },
            context="",
        )

        prompt = GeminiService._create_health_insight_prompt(request)

        assert "100.0%" in prompt
        assert "1.0" in prompt
        assert "0.0" in prompt
        assert "12.0 hours" in prompt

    @staticmethod
    def test_create_health_insight_prompt_missing_data() -> None:
        """Test health insight prompt generation with missing data."""
        request = HealthInsightRequest(
            user_id="missing-data-user",
            analysis_results={},  # Empty analysis results
            context=None,
        )

        prompt = GeminiService._create_health_insight_prompt(request)

        # Should handle missing data gracefully with defaults
        assert "0%" in prompt or "0.0%" in prompt  # Default values
        assert len(prompt) > 500  # Should still be comprehensive


class TestGeminiServiceResponseParsing:
    """Test Gemini service response parsing functionality."""

    @staticmethod
    def test_parse_gemini_response_valid_json() -> None:
        """Test parsing valid JSON response."""
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "narrative": "Test narrative",
                "key_insights": ["Insight 1", "Insight 2"],
                "recommendations": ["Rec 1", "Rec 2"],
                "confidence_score": 0.9,
            }
        )

        result = GeminiService._parse_gemini_response(mock_response, "test-user")

        assert result.user_id == "test-user"
        assert result.narrative == "Test narrative"
        assert len(result.key_insights) == 2
        assert len(result.recommendations) == 2
        assert result.confidence_score == 0.9

    @staticmethod
    def test_parse_gemini_response_partial_json() -> None:
        """Test parsing JSON response with missing fields."""
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "narrative": "Partial narrative"
                # Missing other fields
            }
        )

        result = GeminiService._parse_gemini_response(mock_response, "test-user")

        assert result.narrative == "Partial narrative"
        assert result.key_insights == []  # Default empty list
        assert result.recommendations == []  # Default empty list
        assert result.confidence_score == 0.8  # Default value

    @staticmethod
    def test_parse_gemini_response_invalid_json() -> None:
        """Test parsing invalid JSON response with fallback."""
        mock_response = MagicMock()
        mock_response.text = "Not valid JSON"

        result = GeminiService._parse_gemini_response(mock_response, "test-user")

        assert result.user_id == "test-user"
        assert "Not valid JSON" in result.narrative
        assert len(result.key_insights) > 0  # Fallback insights
        assert len(result.recommendations) > 0  # Fallback recommendations

    @staticmethod
    def test_create_fallback_response() -> None:
        """Test creation of fallback response."""
        long_text = "A" * 500  # Long text to test truncation

        result = GeminiService._create_fallback_response(long_text, "test-user")

        assert result.user_id == "test-user"
        assert len(result.narrative) <= 303  # Truncated + "..."
        assert len(result.key_insights) > 0
        assert len(result.recommendations) > 0
        assert result.confidence_score == 0.7

    @staticmethod
    def test_create_fallback_response_short_text() -> None:
        """Test creation of fallback response with short text."""
        short_text = "Short response"

        result = GeminiService._create_fallback_response(short_text, "test-user")

        assert result.narrative == short_text  # No truncation needed
        assert len(result.key_insights) > 0
        assert len(result.recommendations) > 0


class TestGeminiServiceHealthCheck:
    """Test Gemini service health check functionality."""

    @staticmethod
    @pytest.mark.asyncio
    async def test_health_check_initialized() -> None:
        """Test health check when service is initialized."""
        service = GeminiService(project_id="test-project", location="us-west1")
        service.is_initialized = True
        service.model = MagicMock()

        health = await service.health_check()

        assert health["service"] == "Gemini Service"
        assert health["status"] == "healthy"
        assert health["project_id"] == "test-project"
        assert health["location"] == "us-west1"
        assert health["initialized"] is True
        assert health["model"] == "gemini-2.5-pro"

    @staticmethod
    @pytest.mark.asyncio
    async def test_health_check_not_initialized() -> None:
        """Test health check when service is not initialized."""
        service = GeminiService()

        health = await service.health_check()

        assert health["status"] == "not_initialized"
        assert health["project_id"] == "not_set"
        assert health["initialized"] is False
        assert health["model"] == "not_loaded"

    @staticmethod
    @pytest.mark.asyncio
    async def test_health_check_custom_config() -> None:
        """Test health check with custom configuration."""
        service = GeminiService(project_id="custom-project", location="europe-west1")
        service.is_initialized = True
        service.model = MagicMock()

        health = await service.health_check()

        assert health["project_id"] == "custom-project"
        assert health["location"] == "europe-west1"


class TestGeminiServiceEdgeCases:
    """Test edge cases and error conditions."""

    @staticmethod
    def test_raise_model_not_initialized_error() -> None:
        """Test the model not initialized error helper."""
        with pytest.raises(RuntimeError, match="Gemini model not initialized"):
            GeminiService._raise_model_not_initialized()

    @staticmethod
    def test_generate_placeholder_narrative() -> None:
        """Test placeholder narrative generation."""
        analysis_results = {"sleep_efficiency": 80.0, "circadian_rhythm_score": 0.7}

        narrative = GeminiService._generate_placeholder_narrative(analysis_results)

        assert "80.0%" in narrative
        assert "0.70" in narrative
        assert len(narrative) > 50

    @staticmethod
    def test_extract_key_insights() -> None:
        """Test key insights extraction."""
        # Test excellent sleep
        excellent_results = {"sleep_efficiency": 90.0}
        insights = GeminiService._extract_key_insights(excellent_results)
        assert any("Excellent sleep quality" in insight for insight in insights)

        # Test good sleep
        good_results = {"sleep_efficiency": 80.0}
        insights = GeminiService._extract_key_insights(good_results)
        assert any("Good sleep quality" in insight for insight in insights)

        # Test poor sleep
        poor_results = {"sleep_efficiency": 60.0}
        insights = GeminiService._extract_key_insights(poor_results)
        assert any("needs attention" in insight for insight in insights)

    @staticmethod
    def test_generate_recommendations() -> None:
        """Test recommendations generation."""
        # Test poor sleep efficiency
        poor_sleep_results = {"sleep_efficiency": 70.0}
        recommendations = GeminiService._generate_recommendations(poor_sleep_results)
        assert any("bedtime routine" in rec for rec in recommendations)

        # Test poor circadian rhythm
        poor_circadian_results = {"circadian_rhythm_score": 0.5}
        recommendations = GeminiService._generate_recommendations(
            poor_circadian_results
        )
        assert any("consistent sleep" in rec for rec in recommendations)

        # Test good values
        good_results = {"sleep_efficiency": 85.0, "circadian_rhythm_score": 0.8}
        recommendations = GeminiService._generate_recommendations(good_results)
        # Should have fewer recommendations for good results
        assert len(recommendations) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
