"""Comprehensive tests for Gemini Service.

This test suite covers all aspects of the Gemini service including:
- Service initialization and configuration
- Prompt execution and analysis
- Health check functionality
- Error handling and edge cases
- Integration with PAT analysis results
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from clarity.ml.gemini_service import GeminiService
from clarity.ml.pat_service import ActigraphyAnalysis


class TestGeminiServiceInitialization:
    """Test Gemini service initialization and configuration."""

    def test_service_initialization_default_config(self):
        """Test service initialization with default configuration."""
        service = GeminiService()
        
        assert service.api_key is None
        assert service.model_name == "gemini-2.0-flash-exp"
        assert service.temperature == 0.3
        assert service.max_tokens == 8192
        assert service.client is None

    def test_service_initialization_custom_config(self):
        """Test service initialization with custom configuration."""
        api_key = "test-api-key"
        model_name = "gemini-pro"
        temperature = 0.7
        max_tokens = 4096
        
        service = GeminiService(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        assert service.api_key == api_key
        assert service.model_name == model_name
        assert service.temperature == temperature
        assert service.max_tokens == max_tokens

    def test_service_initialization_from_environment(self):
        """Test service initialization from environment variables."""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'env-api-key'}):
            service = GeminiService()
            assert service.api_key == 'env-api-key'

    def test_client_initialization_with_api_key(self):
        """Test client initialization when API key is provided."""
        with patch('google.generativeai.configure') as mock_configure, \
             patch('google.generativeai.GenerativeModel') as mock_model:
            
            service = GeminiService(api_key="test-key")
            client = service._get_client()
            
            mock_configure.assert_called_once_with(api_key="test-key")
            mock_model.assert_called_once_with("gemini-2.0-flash-exp")
            assert client is not None

    def test_client_initialization_without_api_key(self):
        """Test client initialization fails without API key."""
        service = GeminiService()
        
        with pytest.raises(ValueError, match="Google API key is required"):
            service._get_client()


class TestGeminiServiceAnalysis:
    """Test Gemini service analysis functionality."""

    @pytest.fixture
    def sample_pat_analysis(self) -> ActigraphyAnalysis:
        """Create sample PAT analysis for testing."""
        return ActigraphyAnalysis(
            user_id=str(uuid4()),
            analysis_timestamp="2024-01-01T12:00:00Z",
            sleep_efficiency=85.0,
            sleep_onset_latency=15.0,
            wake_after_sleep_onset=30.0,
            total_sleep_time=7.5,
            circadian_rhythm_score=0.75,
            activity_fragmentation=0.25,
            depression_risk_score=0.2,
            sleep_stages=["wake"] * 100 + ["light_sleep"] * 200 + ["deep_sleep"] * 150,
            confidence_score=0.85,
            clinical_insights=[
                "Good sleep efficiency",
                "Moderate circadian rhythm stability",
                "Low depression risk indicators"
            ]
        )

    @pytest.mark.asyncio
    async def test_analyze_actigraphy_insights_success(self, sample_pat_analysis):
        """Test successful actigraphy insights analysis."""
        service = GeminiService(api_key="test-key")
        
        # Mock the client and response
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "executive_summary": "Excellent sleep patterns with good efficiency",
            "key_findings": [
                "Sleep efficiency of 85% indicates healthy sleep",
                "Low depression risk factors observed"
            ],
            "recommendations": [
                "Maintain current sleep schedule",
                "Continue regular exercise routine"
            ],
            "risk_assessment": {
                "overall_risk": "low",
                "sleep_disorders": "minimal",
                "mental_health": "stable"
            }
        })
        
        mock_client = MagicMock()
        mock_client.generate_content = AsyncMock(return_value=mock_response)
        
        with patch.object(service, '_get_client', return_value=mock_client):
            result = await service.analyze_actigraphy_insights(sample_pat_analysis)
            
            assert "executive_summary" in result
            assert "key_findings" in result
            assert "recommendations" in result
            assert "risk_assessment" in result
            assert result["executive_summary"] == "Excellent sleep patterns with good efficiency"

    @pytest.mark.asyncio
    async def test_analyze_actigraphy_insights_client_error(self, sample_pat_analysis):
        """Test actigraphy insights analysis with client error."""
        service = GeminiService(api_key="test-key")
        
        mock_client = MagicMock()
        mock_client.generate_content = AsyncMock(side_effect=Exception("API Error"))
        
        with patch.object(service, '_get_client', return_value=mock_client):
            with pytest.raises(RuntimeError, match="Failed to analyze actigraphy insights"):
                await service.analyze_actigraphy_insights(sample_pat_analysis)

    @pytest.mark.asyncio
    async def test_analyze_actigraphy_insights_invalid_json(self, sample_pat_analysis):
        """Test actigraphy insights analysis with invalid JSON response."""
        service = GeminiService(api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.text = "Invalid JSON response"
        
        mock_client = MagicMock()
        mock_client.generate_content = AsyncMock(return_value=mock_response)
        
        with patch.object(service, '_get_client', return_value=mock_client):
            with pytest.raises(RuntimeError, match="Failed to parse Gemini response"):
                await service.analyze_actigraphy_insights(sample_pat_analysis)

    @pytest.mark.asyncio
    async def test_execute_prompt_success(self):
        """Test successful prompt execution."""
        service = GeminiService(api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.text = "This is a successful response"
        
        mock_client = MagicMock()
        mock_client.generate_content = AsyncMock(return_value=mock_response)
        
        with patch.object(service, '_get_client', return_value=mock_client):
            result = await service.execute_prompt("Test prompt")
            
            assert result == "This is a successful response"
            mock_client.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_prompt_with_generation_config(self):
        """Test prompt execution with custom generation configuration."""
        service = GeminiService(api_key="test-key", temperature=0.7, max_tokens=4096)
        
        mock_response = MagicMock()
        mock_response.text = "Response with custom config"
        
        mock_client = MagicMock()
        mock_client.generate_content = AsyncMock(return_value=mock_response)
        
        with patch.object(service, '_get_client', return_value=mock_client):
            result = await service.execute_prompt("Test prompt")
            
            assert result == "Response with custom config"
            # Verify generation config was used
            call_args = mock_client.generate_content.call_args
            assert "generation_config" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_execute_prompt_client_error(self):
        """Test prompt execution with client error."""
        service = GeminiService(api_key="test-key")
        
        mock_client = MagicMock()
        mock_client.generate_content = AsyncMock(side_effect=Exception("Client error"))
        
        with patch.object(service, '_get_client', return_value=mock_client):
            with pytest.raises(RuntimeError, match="Failed to execute prompt"):
                await service.execute_prompt("Test prompt")

    @pytest.mark.asyncio
    async def test_execute_prompt_empty_response(self):
        """Test prompt execution with empty response."""
        service = GeminiService(api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.text = ""
        
        mock_client = MagicMock()
        mock_client.generate_content = AsyncMock(return_value=mock_response)
        
        with patch.object(service, '_get_client', return_value=mock_client):
            result = await service.execute_prompt("Test prompt")
            
            assert result == ""


class TestGeminiServicePromptGeneration:
    """Test Gemini service prompt generation functionality."""

    @pytest.fixture
    def sample_pat_analysis(self) -> ActigraphyAnalysis:
        """Create sample PAT analysis for testing."""
        return ActigraphyAnalysis(
            user_id="test-user",
            analysis_timestamp="2024-01-01T12:00:00Z",
            sleep_efficiency=75.0,
            sleep_onset_latency=25.0,
            wake_after_sleep_onset=45.0,
            total_sleep_time=6.5,
            circadian_rhythm_score=0.65,
            activity_fragmentation=0.35,
            depression_risk_score=0.4,
            sleep_stages=["wake"] * 150 + ["light_sleep"] * 200 + ["deep_sleep"] * 100,
            confidence_score=0.75,
            clinical_insights=[
                "Moderate sleep efficiency",
                "Some circadian irregularity",
                "Moderate depression risk indicators"
            ]
        )

    def test_build_actigraphy_prompt_comprehensive(self, sample_pat_analysis):
        """Test comprehensive actigraphy prompt generation."""
        service = GeminiService()
        
        prompt = service._build_actigraphy_prompt(sample_pat_analysis)
        
        # Check that all key components are included
        assert "test-user" in prompt
        assert "75.0%" in prompt  # Sleep efficiency
        assert "25.0 minutes" in prompt  # Sleep onset latency
        assert "6.5 hours" in prompt  # Total sleep time
        assert "0.65" in prompt  # Circadian rhythm score
        assert "0.4" in prompt  # Depression risk score
        assert "Moderate sleep efficiency" in prompt  # Clinical insights
        assert "JSON format" in prompt  # Format specification

    def test_build_actigraphy_prompt_edge_cases(self):
        """Test actigraphy prompt generation with edge case values."""
        analysis = ActigraphyAnalysis(
            user_id="edge-case-user",
            analysis_timestamp="2024-01-01T12:00:00Z",
            sleep_efficiency=100.0,  # Perfect efficiency
            sleep_onset_latency=0.0,  # Instant sleep
            wake_after_sleep_onset=0.0,  # No awakenings
            total_sleep_time=12.0,  # Long sleep
            circadian_rhythm_score=1.0,  # Perfect rhythm
            activity_fragmentation=0.0,  # No fragmentation
            depression_risk_score=0.0,  # No risk
            sleep_stages=["deep_sleep"] * 720,  # All deep sleep
            confidence_score=1.0,  # Perfect confidence
            clinical_insights=["Perfect sleep patterns"]
        )
        
        service = GeminiService()
        prompt = service._build_actigraphy_prompt(analysis)
        
        assert "100.0%" in prompt
        assert "0.0 minutes" in prompt
        assert "12.0 hours" in prompt
        assert "1.0" in prompt
        assert "Perfect sleep patterns" in prompt

    def test_build_actigraphy_prompt_empty_insights(self):
        """Test actigraphy prompt generation with empty clinical insights."""
        analysis = ActigraphyAnalysis(
            user_id="no-insights-user",
            analysis_timestamp="2024-01-01T12:00:00Z",
            sleep_efficiency=80.0,
            sleep_onset_latency=20.0,
            wake_after_sleep_onset=40.0,
            total_sleep_time=7.0,
            circadian_rhythm_score=0.7,
            activity_fragmentation=0.3,
            depression_risk_score=0.3,
            sleep_stages=["wake", "light_sleep", "deep_sleep"],
            confidence_score=0.8,
            clinical_insights=[]  # Empty insights
        )
        
        service = GeminiService()
        prompt = service._build_actigraphy_prompt(analysis)
        
        # Should still generate a valid prompt
        assert "no-insights-user" in prompt
        assert "80.0%" in prompt
        assert len(prompt) > 500  # Should still be comprehensive


class TestGeminiServiceHealthCheck:
    """Test Gemini service health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_with_api_key(self):
        """Test health check when API key is configured."""
        service = GeminiService(api_key="test-key")
        
        health = await service.health_check()
        
        assert health["service"] == "Gemini LLM Service"
        assert health["status"] == "healthy"
        assert health["model_name"] == "gemini-2.0-flash-exp"
        assert health["api_key_configured"] is True

    @pytest.mark.asyncio
    async def test_health_check_without_api_key(self):
        """Test health check when API key is not configured."""
        service = GeminiService()
        
        health = await service.health_check()
        
        assert health["service"] == "Gemini LLM Service"
        assert health["status"] == "configuration_needed"
        assert health["api_key_configured"] is False

    @pytest.mark.asyncio
    async def test_health_check_custom_model(self):
        """Test health check with custom model configuration."""
        service = GeminiService(
            api_key="test-key",
            model_name="gemini-pro",
            temperature=0.5,
            max_tokens=2048
        )
        
        health = await service.health_check()
        
        assert health["model_name"] == "gemini-pro"
        assert health["temperature"] == 0.5
        assert health["max_tokens"] == 2048


class TestGeminiServiceEdgeCases:
    """Test edge cases and error conditions."""

    def test_get_client_caching(self):
        """Test that client is cached after first creation."""
        with patch('google.generativeai.configure') as mock_configure, \
             patch('google.generativeai.GenerativeModel') as mock_model:
            
            service = GeminiService(api_key="test-key")
            
            # First call should create client
            client1 = service._get_client()
            # Second call should return cached client
            client2 = service._get_client()
            
            assert client1 is client2
            # Configure should only be called once
            mock_configure.assert_called_once()
            mock_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_actigraphy_insights_no_api_key(self):
        """Test analysis fails without API key."""
        service = GeminiService()
        
        analysis = ActigraphyAnalysis(
            user_id="test",
            analysis_timestamp="2024-01-01T12:00:00Z",
            sleep_efficiency=80.0,
            sleep_onset_latency=20.0,
            wake_after_sleep_onset=40.0,
            total_sleep_time=7.0,
            circadian_rhythm_score=0.7,
            activity_fragmentation=0.3,
            depression_risk_score=0.3,
            sleep_stages=["wake"],
            confidence_score=0.8,
            clinical_insights=["Test insight"]
        )
        
        with pytest.raises(ValueError, match="Google API key is required"):
            await service.analyze_actigraphy_insights(analysis)

    @pytest.mark.asyncio
    async def test_execute_prompt_no_api_key(self):
        """Test prompt execution fails without API key."""
        service = GeminiService()
        
        with pytest.raises(ValueError, match="Google API key is required"):
            await service.execute_prompt("Test prompt")

    def test_build_prompt_handles_none_values(self):
        """Test prompt building handles None values gracefully."""
        analysis = ActigraphyAnalysis(
            user_id="test-user",
            analysis_timestamp="2024-01-01T12:00:00Z",
            sleep_efficiency=80.0,
            sleep_onset_latency=20.0,
            wake_after_sleep_onset=40.0,
            total_sleep_time=7.0,
            circadian_rhythm_score=0.7,
            activity_fragmentation=0.3,
            depression_risk_score=0.3,
            sleep_stages=[],  # Empty list
            confidence_score=0.8,
            clinical_insights=[]  # Empty list
        )
        
        service = GeminiService()
        prompt = service._build_actigraphy_prompt(analysis)
        
        # Should not crash and should produce valid prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "test-user" in prompt

    @pytest.mark.asyncio
    async def test_analyze_insights_malformed_json_fallback(self):
        """Test analysis handles malformed JSON by falling back to text."""
        service = GeminiService(api_key="test-key")
        
        analysis = ActigraphyAnalysis(
            user_id="test",
            analysis_timestamp="2024-01-01T12:00:00Z",
            sleep_efficiency=80.0,
            sleep_onset_latency=20.0,
            wake_after_sleep_onset=40.0,
            total_sleep_time=7.0,
            circadian_rhythm_score=0.7,
            activity_fragmentation=0.3,
            depression_risk_score=0.3,
            sleep_stages=["wake"],
            confidence_score=0.8,
            clinical_insights=["Test insight"]
        )
        
        # Mock response with malformed JSON
        mock_response = MagicMock()
        mock_response.text = '{"incomplete": json'
        
        mock_client = MagicMock()
        mock_client.generate_content = AsyncMock(return_value=mock_response)
        
        with patch.object(service, '_get_client', return_value=mock_client):
            with pytest.raises(RuntimeError, match="Failed to parse Gemini response"):
                await service.analyze_actigraphy_insights(analysis)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 