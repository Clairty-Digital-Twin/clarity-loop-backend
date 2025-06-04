"""Tests for insight subscriber functionality."""

import asyncio
import base64
import json
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException, Request
import pytest

from clarity.ml.gemini_service import HealthInsightResponse
from clarity.services.pubsub.insight_subscriber import (
    HIGH_CONSISTENCY_THRESHOLD,
    MIN_CARDIO_FEATURES_REQUIRED,
    MIN_FEATURE_VECTOR_LENGTH,
    MIN_RESPIRATORY_FEATURES_REQUIRED,
    MODERATE_CONSISTENCY_THRESHOLD,
    GeminiInsightGenerator,
    InsightSubscriber,
    InsightSubscriberSingleton,
    generate_insight_task,
    get_insight_subscriber,
    health_check,
)


class TestConstants:
    """🔢 Test module constants."""

    @staticmethod
    def test_constants_values() -> None:
        """Test that constants have correct values."""
        assert HIGH_CONSISTENCY_THRESHOLD == 0.8
        assert MODERATE_CONSISTENCY_THRESHOLD == 0.5
        assert MIN_FEATURE_VECTOR_LENGTH == 8
        assert MIN_CARDIO_FEATURES_REQUIRED == 3
        assert MIN_RESPIRATORY_FEATURES_REQUIRED == 4


class TestGeminiInsightGeneratorInitialization:
    """🔧 Test GeminiInsightGenerator initialization."""

    @staticmethod
    @patch("clarity.services.pubsub.insight_subscriber.GeminiService")
    @patch("clarity.services.pubsub.insight_subscriber.FirestoreClient")
    def test_init_with_api_key(
        mock_firestore_class: Mock, mock_gemini_class: Mock
    ) -> None:
        """Test initialization with proper dependencies."""
        mock_gemini_service = Mock()
        mock_gemini_class.return_value = mock_gemini_service
        mock_firestore_client = Mock()
        mock_firestore_class.return_value = mock_firestore_client

        generator = GeminiInsightGenerator(project_id="test-project")

        assert generator.gemini_service == mock_gemini_service
        assert generator.firestore_client == mock_firestore_client
        mock_gemini_class.assert_called_once_with(project_id="test-project")
        mock_firestore_class.assert_called_once_with(project_id="test-project")

    @staticmethod
    @patch("clarity.services.pubsub.insight_subscriber.GeminiService")
    @patch("clarity.services.pubsub.insight_subscriber.FirestoreClient")
    def test_init_without_api_key(
        mock_firestore_class: Mock, mock_gemini_class: Mock
    ) -> None:
        """Test initialization with default project."""
        mock_gemini_service = Mock()
        mock_gemini_class.return_value = mock_gemini_service
        mock_firestore_client = Mock()
        mock_firestore_class.return_value = mock_firestore_client

        generator = GeminiInsightGenerator()

        assert generator.gemini_service == mock_gemini_service
        assert generator.firestore_client == mock_firestore_client
        mock_gemini_class.assert_called_once_with(project_id=None)
        mock_firestore_class.assert_called_once_with(project_id="clarity-digital-twin")


@pytest.fixture
def mock_generator() -> GeminiInsightGenerator:
    """Create GeminiInsightGenerator with mocks."""
    with (
        patch(
            "clarity.services.pubsub.insight_subscriber.GeminiService"
        ) as mock_gemini_class,
        patch(
            "clarity.services.pubsub.insight_subscriber.FirestoreClient"
        ) as mock_firestore_class,
    ):
        mock_gemini_service = Mock()
        mock_gemini_class.return_value = mock_gemini_service
        mock_firestore_client = Mock()
        mock_firestore_class.return_value = mock_firestore_client

        generator = GeminiInsightGenerator()
        generator.gemini_service = mock_gemini_service
        generator.firestore_client = mock_firestore_client
        return generator


@pytest.fixture
def sample_analysis_results() -> dict[str, Any]:
    """Sample analysis results for testing."""
    return {
        "cardio_features": [75.0, 150.0, 65.0, 0.0, 45.2, 0.0, 0.85, 0.92],
        "respiratory_features": [16.5, 14.2, 0.0, 98.5, 95.2, 0.0, 0.88, 0.91],
        "sleep_features": {
            "sleep_efficiency": 0.85,
            "total_sleep_minutes": 480,
            "waso_minutes": 30,
            "sleep_latency": 15,
            "rem_percentage": 0.22,
            "deep_percentage": 0.18,
            "consistency_score": 0.75,
        },
        "summary_stats": {
            "health_indicators": {
                "cardiovascular_health": {"circadian_rhythm": 0.88},
                "respiratory_health": {"respiratory_stability": 0.91},
            }
        },
    }


class TestGeminiInsightGeneratorAnalysisEnhancement:
    """🔬 Test analysis results enhancement."""

    @staticmethod
    def test_enhance_analysis_results_for_gemini_basic(
        sample_analysis_results: dict[str, Any],
    ) -> None:
        """Test basic enhancement of analysis results."""
        enhanced = GeminiInsightGenerator._enhance_analysis_results_for_gemini(
            sample_analysis_results
        )

        # Check sleep metrics mapping
        assert enhanced["sleep_efficiency"] == 85.0  # 0.85 * 100
        assert enhanced["total_sleep_time"] == 8.0  # 480 / 60
        assert enhanced["wake_after_sleep_onset"] == 30
        assert enhanced["sleep_onset_latency"] == 15
        assert enhanced["rem_sleep_percent"] == 22.0  # 0.22 * 100
        assert enhanced["deep_sleep_percent"] == 18.0  # 0.18 * 100
        assert enhanced["sleep_consistency_rating"] == "moderate"  # 0.75

    @staticmethod
    def test_enhance_analysis_results_sleep_consistency_high() -> None:
        """Test sleep consistency rating - high."""
        analysis_results = {
            "sleep_features": {
                "consistency_score": 0.85,  # > 0.8
                "sleep_efficiency": 0.9,
                "total_sleep_minutes": 420,
                "waso_minutes": 20,
                "sleep_latency": 10,
                "rem_percentage": 0.25,
                "deep_percentage": 0.20,
            }
        }

        enhanced = GeminiInsightGenerator._enhance_analysis_results_for_gemini(
            analysis_results
        )

        assert enhanced["sleep_consistency_rating"] == "high"

    @staticmethod
    def test_enhance_analysis_results_sleep_consistency_low() -> None:
        """Test sleep consistency rating - low."""
        analysis_results = {
            "sleep_features": {
                "consistency_score": 0.3,  # <= 0.5
                "sleep_efficiency": 0.7,
                "total_sleep_minutes": 360,
                "waso_minutes": 60,
                "sleep_latency": 30,
                "rem_percentage": 0.15,
                "deep_percentage": 0.12,
            }
        }

        enhanced = GeminiInsightGenerator._enhance_analysis_results_for_gemini(
            analysis_results
        )

        assert enhanced["sleep_consistency_rating"] == "low"

    @staticmethod
    def test_enhance_analysis_results_no_sleep_features() -> None:
        """Test enhancement when no sleep features present."""
        analysis_results = {"cardio_features": [75.0, 150.0, 65.0]}

        enhanced = GeminiInsightGenerator._enhance_analysis_results_for_gemini(
            analysis_results
        )

        # Should not have sleep metrics added
        assert "sleep_efficiency" not in enhanced
        assert "sleep_consistency_rating" not in enhanced

    @staticmethod
    def test_enhance_analysis_results_pydantic_model() -> None:
        """Test enhancement with Pydantic model sleep features."""
        # Mock Pydantic model
        mock_sleep_features = Mock()
        mock_sleep_features.model_dump.return_value = {
            "sleep_efficiency": 0.88,
            "total_sleep_minutes": 450,
            "consistency_score": 0.6,
        }

        analysis_results = {"sleep_features": mock_sleep_features}

        enhanced = GeminiInsightGenerator._enhance_analysis_results_for_gemini(
            analysis_results
        )

        assert enhanced["sleep_efficiency"] == 88.0
        assert enhanced["total_sleep_time"] == 7.5

    @staticmethod
    def test_enhance_analysis_results_old_pydantic_model() -> None:
        """Test enhancement with old Pydantic model (dict method)."""
        # Mock old Pydantic model that doesn't have model_dump but has dict()
        mock_sleep_features = Mock()
        mock_sleep_features.model_dump = Mock(
            side_effect=AttributeError("No model_dump")
        )
        mock_sleep_features.dict = Mock(
            return_value={
                "sleep_efficiency": 0.82,
                "total_sleep_minutes": 420,
                "consistency_score": 0.7,
            }
        )

        analysis_results = {"sleep_features": mock_sleep_features}

        enhanced = GeminiInsightGenerator._enhance_analysis_results_for_gemini(
            analysis_results
        )

        assert enhanced["sleep_efficiency"] == 82.0
        mock_sleep_features.dict.assert_called_once()


class TestGeminiInsightGeneratorPromptCreation:
    """📝 Test health prompt creation."""

    @staticmethod
    def test_create_health_prompt_full_metrics(
        sample_analysis_results: dict[str, Any],
    ) -> None:
        """Test prompt creation with full metrics."""
        prompt = GeminiInsightGenerator._create_health_prompt(sample_analysis_results)

        # Check that enhanced metrics are included
        assert "Sleep Efficiency: 85.0%" in prompt
        assert "Total Sleep Time: 8.0 hours" in prompt
        assert "Average Heart Rate: 75.0 bpm" in prompt
        assert "Max Heart Rate: 150.0 bpm" in prompt
        assert "Min Heart Rate: 65.0 bpm" in prompt
        assert "Average Respiratory Rate: 16.5 rpm" in prompt
        assert "SpO2 Average: 98.5%" in prompt
        assert "Circadian Rhythm Score: 0.88/1.0" in prompt
        assert "Respiratory Health Score: 0.91/1.0" in prompt

    @staticmethod
    def test_create_health_prompt_with_sleep_metrics() -> None:
        """Test prompt creation with sleep metrics."""
        analysis_results = {
            "sleep_features": {
                "sleep_efficiency": 0.9,
                "total_sleep_minutes": 480,
                "consistency_score": 0.85,
            }
        }

        prompt = GeminiInsightGenerator._create_health_prompt(analysis_results)

        assert "Sleep Efficiency: 90.0%" in prompt
        assert "Total Sleep Time: 8.0 hours" in prompt
        assert "Sleep Consistency: High" in prompt

    @staticmethod
    def test_create_health_prompt_insufficient_cardio_features() -> None:
        """Test prompt creation with insufficient cardio features."""
        analysis_results = {"cardio_features": [75.0, 150.0]}  # Only 2 features, need 3

        prompt = GeminiInsightGenerator._create_health_prompt(analysis_results)

        assert "Average Heart Rate" not in prompt

    @staticmethod
    def test_create_health_prompt_no_features() -> None:
        """Test prompt creation with no features."""
        analysis_results: dict[str, Any] = {}

        prompt = GeminiInsightGenerator._create_health_prompt(analysis_results)

        assert "health metrics" in prompt.lower()


class TestGeminiInsightGeneratorModelNotInitializedError:
    """💥 Test model not initialized error."""

    @staticmethod
    def test_raise_model_not_initialized_error() -> None:
        """Test raising model not initialized error."""
        with pytest.raises(RuntimeError, match="Gemini model not properly initialized"):
            GeminiInsightGenerator._raise_model_not_initialized_error()


class TestGeminiInsightGeneratorInsightGeneration:
    """🧠 Test insight generation functionality."""

    @staticmethod
    async def test_generate_health_insight_with_model(
        mock_generator: GeminiInsightGenerator, sample_analysis_results: dict[str, Any]
    ) -> None:
        """Test insight generation with GeminiService."""
        user_id = "test-user-123"
        upload_id = "upload-456"

        # Mock GeminiService response structure
        mock_response = HealthInsightResponse(
            key_insights=["Great sleep efficiency!", "Heart rate variability is good."],
            recommendations=[
                "Continue current sleep routine",
                "Consider morning exercise",
            ],
            confidence_score=0.85,
            narrative="Overall health looks good with room for improvement.",
            generated_at="2024-01-01T00:00:00Z",
            user_id=user_id,
        )

        with (
            patch.object(
                mock_generator.gemini_service,
                "generate_health_insights",
                new=AsyncMock(return_value=mock_response),
            ) as mock_gemini,
            patch.object(
                mock_generator, "_store_insight", new=AsyncMock()
            ) as mock_store,
        ):
            result = await mock_generator.generate_health_insight(
                user_id, upload_id, sample_analysis_results
            )

            assert result["status"] == "success"
            assert result["message"] == "Health insight generated successfully"
            assert result["insights"] == mock_response.key_insights
            assert result["recommendations"] == mock_response.recommendations
            assert result["health_score"] == 85  # 0.85 * 100
            assert result["confidence_level"] == "high"
            assert result["narrative"] == mock_response.narrative

            mock_gemini.assert_called_once()
            mock_store.assert_called_once()

    @staticmethod
    async def test_generate_health_insight_without_model(
        mock_generator: GeminiInsightGenerator, sample_analysis_results: dict[str, Any]
    ) -> None:
        """Test insight generation with fallback when GeminiService fails."""
        user_id = "test-user-123"
        upload_id = "upload-456"

        # Mock GeminiService to fail and trigger fallback
        with (
            patch.object(
                mock_generator.gemini_service,
                "generate_health_insights",
                new=AsyncMock(side_effect=Exception("Service unavailable")),
            ),
            patch.object(
                mock_generator, "_store_insight", new=AsyncMock()
            ) as mock_store,
        ):
            result = await mock_generator.generate_health_insight(
                user_id, upload_id, sample_analysis_results
            )

            # Should return fallback insight
            assert result["status"] == "success"
            assert result["message"] == "Health insight generated successfully"
            assert "insights" in result
            assert "recommendations" in result
            assert "health_score" in result
            assert isinstance(result["insights"], list)

            mock_store.assert_called_once()

    @staticmethod
    async def test_generate_health_insight_exception(
        mock_generator: GeminiInsightGenerator, sample_analysis_results: dict[str, Any]
    ) -> None:
        """Test insight generation with exception in storage."""
        user_id = "test-user-123"
        upload_id = "upload-456"

        with (
            patch.object(
                mock_generator.gemini_service,
                "generate_health_insights",
                new=AsyncMock(side_effect=Exception("Service error")),
            ),
            patch.object(
                mock_generator,
                "_store_insight",
                new=AsyncMock(side_effect=Exception("Storage error")),
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await mock_generator.generate_health_insight(
                user_id, upload_id, sample_analysis_results
            )

        assert exc_info.value.status_code == 500


class TestInsightSubscriberInitialization:
    """🔧 Test InsightSubscriber initialization."""

    @staticmethod
    @patch("clarity.services.pubsub.insight_subscriber.GeminiInsightGenerator")
    def test_insight_subscriber_init(mock_generator_class: Mock) -> None:
        """Test InsightSubscriber initialization."""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        subscriber = InsightSubscriber()

        assert subscriber.generator == mock_generator
        mock_generator_class.assert_called_once_with(project_id=None)


@pytest.fixture
def insight_subscriber() -> InsightSubscriber:
    """Create InsightSubscriber with mocked generator."""
    with patch("clarity.services.pubsub.insight_subscriber.GeminiInsightGenerator"):
        return InsightSubscriber()


class TestInsightSubscriberMessageProcessing:
    """📨 Test message processing functionality."""

    @staticmethod
    async def test_process_insight_request_message_success(
        insight_subscriber: InsightSubscriber,
    ) -> None:
        """Test successful message processing."""
        # Create mock request
        mock_request = Mock(spec=Request)

        # Create proper Pub/Sub message structure
        message_data = {
            "user_id": "test-user-123",
            "upload_id": "upload-456",
            "analysis_results": {"features": [1, 2, 3]},
        }
        encoded_data = base64.b64encode(json.dumps(message_data).encode()).decode()

        pubsub_body = {
            "message": {
                "data": encoded_data,
                "messageId": "test-message-123",
                "publishTime": "2023-01-01T00:00:00Z",
            }
        }

        mock_request.json = AsyncMock(return_value=pubsub_body)

        # Mock the generate_health_insight method
        mock_insight = {
            "status": "success",
            "message": "Health insight generated successfully",
            "insights": ["test"],
            "health_score": 85,
        }

        with patch.object(
            insight_subscriber.generator,
            "generate_health_insight",
            new=AsyncMock(return_value=mock_insight),
        ) as mock_generate:
            result = await insight_subscriber.process_insight_request_message(
                mock_request
            )

            assert result["status"] == "success"
            assert result["message"] == "Health insight generated successfully"

            mock_generate.assert_called_once_with(
                user_id="test-user-123",
                upload_id="upload-456",
                analysis_results={"features": [1, 2, 3]},
            )

    @staticmethod
    async def test_process_insight_request_message_missing_fields(
        insight_subscriber: InsightSubscriber,
    ) -> None:
        """Test message processing with missing required fields."""
        mock_request = Mock(spec=Request)

        # Missing user_id
        message_data = {
            "upload_id": "upload-456",
            "analysis_results": {"features": [1, 2, 3]},
        }
        encoded_data = base64.b64encode(json.dumps(message_data).encode()).decode()

        pubsub_body = {"message": {"data": encoded_data}}

        mock_request.json = AsyncMock(return_value=pubsub_body)

        with pytest.raises(HTTPException) as exc_info:
            await insight_subscriber.process_insight_request_message(mock_request)

        assert exc_info.value.status_code == 400
        assert "user_id" in str(exc_info.value.detail)

    @staticmethod
    async def test_process_insight_request_message_invalid_base64(
        insight_subscriber: InsightSubscriber,
    ) -> None:
        """Test message processing with invalid base64 data."""
        mock_request = Mock(spec=Request)

        pubsub_body = {"message": {"data": "invalid-base64!@#$"}}

        mock_request.json = AsyncMock(return_value=pubsub_body)

        with pytest.raises(HTTPException) as exc_info:
            await insight_subscriber.process_insight_request_message(mock_request)

        assert exc_info.value.status_code == 400

    @staticmethod
    async def test_process_insight_request_message_invalid_json(
        insight_subscriber: InsightSubscriber,
    ) -> None:
        """Test message processing with invalid JSON in data."""
        mock_request = Mock(spec=Request)

        # Valid base64 but invalid JSON
        encoded_data = base64.b64encode(b"invalid json").decode()

        pubsub_body = {"message": {"data": encoded_data}}

        mock_request.json = AsyncMock(return_value=pubsub_body)

        with pytest.raises(HTTPException) as exc_info:
            await insight_subscriber.process_insight_request_message(mock_request)

        assert exc_info.value.status_code == 400

    @staticmethod
    async def test_process_insight_request_message_generation_error(
        insight_subscriber: InsightSubscriber,
    ) -> None:
        """Test message processing with insight generation error."""
        mock_request = Mock(spec=Request)

        message_data = {
            "user_id": "test-user-123",
            "upload_id": "upload-456",
            "analysis_results": {"features": [1, 2, 3]},
        }
        encoded_data = base64.b64encode(json.dumps(message_data).encode()).decode()

        pubsub_body = {"message": {"data": encoded_data}}

        mock_request.json = AsyncMock(return_value=pubsub_body)

        # Mock generation failure
        with patch.object(
            insight_subscriber.generator,
            "generate_health_insight",
            new=AsyncMock(side_effect=Exception("Generation failed")),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await insight_subscriber.process_insight_request_message(mock_request)

            assert exc_info.value.status_code == 500


class TestInsightSubscriberSingleton:
    """🎯 Test InsightSubscriberSingleton."""

    @staticmethod
    def test_singleton_get_instance_first_call() -> None:
        """Test singleton first call creates instance."""
        # Reset singleton
        InsightSubscriberSingleton._instance = None

        with patch(
            "clarity.services.pubsub.insight_subscriber.InsightSubscriber"
        ) as mock_subscriber_class:
            mock_instance = Mock()
            mock_subscriber_class.return_value = mock_instance

            result = InsightSubscriberSingleton.get_instance()

            assert result == mock_instance
            mock_subscriber_class.assert_called_once()

    @staticmethod
    def test_singleton_get_instance_subsequent_calls() -> None:
        """Test singleton subsequent calls return same instance."""
        # Set up existing instance
        existing_instance = Mock()
        InsightSubscriberSingleton._instance = existing_instance

        with patch(
            "clarity.services.pubsub.insight_subscriber.InsightSubscriber"
        ) as mock_subscriber_class:
            result1 = InsightSubscriberSingleton.get_instance()
            result2 = InsightSubscriberSingleton.get_instance()

            assert result1 == existing_instance
            assert result2 == existing_instance
            assert result1 is result2
            mock_subscriber_class.assert_not_called()


class TestInsightSubscriberGlobalFunction:
    """🌍 Test global function."""

    @staticmethod
    def test_get_insight_subscriber() -> None:
        """Test global get_insight_subscriber function."""
        with patch.object(
            InsightSubscriberSingleton, "get_instance"
        ) as mock_get_instance:
            mock_instance = Mock()
            mock_get_instance.return_value = mock_instance

            result = get_insight_subscriber()

            assert result == mock_instance
            mock_get_instance.assert_called_once()


class TestEdgeCasesAndBoundaryConditions:
    """🎯 Test edge cases and boundary conditions."""

    @staticmethod
    def test_empty_analysis_results() -> None:
        """Test handling of empty analysis results."""
        enhanced = GeminiInsightGenerator._enhance_analysis_results_for_gemini({})
        assert isinstance(enhanced, dict)

    @staticmethod
    def test_malformed_sleep_features() -> None:
        """Test handling of malformed sleep features."""
        analysis_results = {"sleep_features": "not_a_dict"}  # Invalid type

        # Should not raise an exception and return a valid dict
        enhanced = GeminiInsightGenerator._enhance_analysis_results_for_gemini(
            analysis_results
        )
        assert isinstance(enhanced, dict)

    @staticmethod
    def test_partial_feature_vectors() -> None:
        """Test handling of partial feature vectors."""
        analysis_results = {
            "cardio_features": [75.0, 150.0]  # Only 2 features, less than minimum
        }

        prompt = GeminiInsightGenerator._create_health_prompt(analysis_results)

        # Should not include heart rate metrics due to insufficient features
        assert "Average Heart Rate" not in prompt

    @staticmethod
    async def test_concurrent_insight_generation(
        mock_generator: GeminiInsightGenerator, sample_analysis_results: dict[str, Any]
    ) -> None:
        """Test concurrent insight generation."""
        # Mock fallback mode for speed and consistency
        with (
            patch.object(
                mock_generator.gemini_service,
                "generate_health_insights",
                new=AsyncMock(side_effect=Exception("Service unavailable")),
            ),
            patch.object(mock_generator, "_store_insight", new=AsyncMock()),
        ):

            async def generate_insight(i: int) -> dict[str, Any]:
                return await mock_generator.generate_health_insight(
                    f"user-{i}", f"upload-{i}", sample_analysis_results
                )

            # Test concurrent generation
            tasks = [generate_insight(i) for i in range(3)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 3
            for result in results:
                assert result["status"] == "success"
                assert "insights" in result
                assert "health_score" in result
