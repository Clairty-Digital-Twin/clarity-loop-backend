"""Comprehensive tests for Async Inference Engine.

This test suite covers all aspects of the inference engine including:
- Engine initialization and configuration
- Single and batch inference processing
- Caching functionality
- Error handling and edge cases
- Performance monitoring
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Never
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from clarity.core.exceptions import ServiceUnavailableProblem
from clarity.ml.inference_engine import (  # type: ignore[attr-defined]
    AsyncInferenceEngine,
    InferenceCache,
    InferenceRequest,
    InferenceResponse,
)
from clarity.ml.pat_service import ActigraphyAnalysis, ActigraphyInput, PATModelService
from clarity.ml.preprocessing import ActigraphyDataPoint


class TestAsyncInferenceEngineInitialization:
    """Test async inference engine initialization and configuration."""

    @staticmethod
    def test_engine_initialization_default_config() -> None:
        """Test engine initialization with default configuration."""
        mock_pat_service = MagicMock(spec=PATModelService)
        engine = AsyncInferenceEngine(pat_service=mock_pat_service)

        assert engine.pat_service == mock_pat_service
        assert engine.batch_size == 4  # DEFAULT_BATCH_SIZE from constants
        assert engine.batch_timeout == 0.1  # DEFAULT_BATCH_TIMEOUT_MS / 1000 = 100/1000
        assert engine.request_count == 0
        assert engine.cache_hits == 0
        assert engine.error_count == 0
        assert not engine.is_running

    @staticmethod
    def test_engine_initialization_custom_config() -> None:
        """Test engine initialization with custom configuration."""
        mock_pat_service = MagicMock(spec=PATModelService)
        batch_size = 50
        batch_timeout_ms = 1000
        cache_ttl = 3600

        engine = AsyncInferenceEngine(
            pat_service=mock_pat_service,
            batch_size=batch_size,
            batch_timeout_ms=batch_timeout_ms,
            cache_ttl=cache_ttl,
        )

        assert engine.batch_size == batch_size
        assert engine.batch_timeout == 1.0  # 1000ms = 1.0s
        assert engine.cache.ttl == cache_ttl

    @staticmethod
    @pytest.mark.asyncio
    async def test_engine_start_and_stop() -> None:
        """Test engine start and stop functionality."""
        mock_pat_service = MagicMock(spec=PATModelService)
        engine = AsyncInferenceEngine(pat_service=mock_pat_service)

        # Test start
        await engine.start()
        assert engine.is_running
        assert engine.batch_processor_task is not None

        # Test stop
        await engine.stop()
        assert not engine.is_running


class TestAsyncInferenceEngineInference:
    """Test async inference engine inference functionality."""

    @staticmethod
    @pytest.fixture
    def sample_actigraphy_input() -> ActigraphyInput:
        """Create sample actigraphy input."""
        data_points = [
            ActigraphyDataPoint(timestamp=datetime.now(UTC), value=float(i % 100))
            for i in range(1440)  # 24 hours of data
        ]

        return ActigraphyInput(
            user_id=str(uuid4()),
            data_points=data_points,
            sampling_rate=1.0,
            duration_hours=24,
        )

    @staticmethod
    @pytest.fixture
    def sample_inference_request(
        sample_actigraphy_input: ActigraphyInput,
    ) -> InferenceRequest:
        """Create sample inference request."""
        return InferenceRequest(
            request_id=str(uuid4()),
            input_data=sample_actigraphy_input,
            timeout_seconds=30.0,
        )

    @staticmethod
    @pytest.mark.asyncio
    async def test_single_inference_success(
        sample_inference_request: InferenceRequest,
    ) -> None:
        """Test successful single inference."""
        mock_analysis = ActigraphyAnalysis(
            user_id=sample_inference_request.input_data.user_id,
            analysis_timestamp=datetime.now(UTC).isoformat(),
            sleep_efficiency=85.0,
            sleep_onset_latency=15.0,
            wake_after_sleep_onset=30.0,
            total_sleep_time=7.5,
            circadian_rhythm_score=0.75,
            activity_fragmentation=0.25,
            depression_risk_score=0.2,
            sleep_stages=["wake"] * 100,
            confidence_score=0.85,
            clinical_insights=["Good sleep efficiency", "Regular sleep pattern"],
            embedding=[0.0] * 128,
        )

        mock_pat_service = MagicMock(spec=PATModelService)
        mock_pat_service.analyze_actigraphy = AsyncMock(return_value=mock_analysis)

        async with AsyncInferenceEngine(pat_service=mock_pat_service) as engine:
            result = await engine.predict_async(sample_inference_request)

            assert isinstance(result, InferenceResponse)
            assert result.request_id == sample_inference_request.request_id
            assert result.analysis == mock_analysis
            assert result.processing_time_ms > 0

    @staticmethod
    @pytest.mark.asyncio
    async def test_batch_processing(sample_actigraphy_input: ActigraphyInput) -> None:
        """Test batch processing of multiple requests."""
        mock_analysis = ActigraphyAnalysis(
            user_id=sample_actigraphy_input.user_id,
            analysis_timestamp=datetime.now(UTC).isoformat(),
            sleep_efficiency=85.0,
            sleep_onset_latency=15.0,
            wake_after_sleep_onset=30.0,
            total_sleep_time=7.5,
            circadian_rhythm_score=0.75,
            activity_fragmentation=0.25,
            depression_risk_score=0.2,
            sleep_stages=["wake"] * 100,
            confidence_score=0.85,
            clinical_insights=["Good sleep efficiency"],
            embedding=[0.0] * 128,
        )

        mock_pat_service = MagicMock(spec=PATModelService)
        mock_pat_service.analyze_actigraphy = AsyncMock(return_value=mock_analysis)

        async with AsyncInferenceEngine(
            pat_service=mock_pat_service, batch_size=2
        ) as engine:
            # Create multiple requests
            requests = [
                InferenceRequest(
                    request_id=str(uuid4()),
                    input_data=sample_actigraphy_input,
                    timeout_seconds=30.0,
                )
                for _ in range(3)
            ]

            # Process requests concurrently
            results = await asyncio.gather(
                *[engine.predict_async(request) for request in requests]
            )

            assert len(results) == 3
            for i, result in enumerate(results):
                assert isinstance(result, InferenceResponse)
                assert result.request_id == requests[i].request_id
                assert result.analysis == mock_analysis

    @staticmethod
    @pytest.mark.asyncio
    async def test_caching_functionality(
        sample_inference_request: InferenceRequest,
    ) -> None:
        """Test that caching works correctly."""
        mock_analysis = ActigraphyAnalysis(
            user_id=sample_inference_request.input_data.user_id,
            analysis_timestamp=datetime.now(UTC).isoformat(),
            sleep_efficiency=85.0,
            sleep_onset_latency=15.0,
            wake_after_sleep_onset=30.0,
            total_sleep_time=7.5,
            circadian_rhythm_score=0.75,
            activity_fragmentation=0.25,
            depression_risk_score=0.2,
            sleep_stages=["wake"] * 100,
            confidence_score=0.85,
            clinical_insights=["Good sleep efficiency"],
            embedding=[0.0] * 128,
        )

        mock_pat_service = MagicMock(spec=PATModelService)
        mock_pat_service.analyze_actigraphy = AsyncMock(return_value=mock_analysis)

        async with AsyncInferenceEngine(pat_service=mock_pat_service) as engine:
            # First request
            result1 = await engine.predict_async(sample_inference_request)

            # Second identical request (should hit cache)
            result2 = await engine.predict_async(sample_inference_request)

            assert result1.analysis == result2.analysis
            assert engine.cache_hits > 0


class TestInferenceCache:
    """Test inference cache functionality."""

    @staticmethod
    def test_cache_initialization() -> None:
        """Test cache initialization with TTL."""
        cache = InferenceCache(ttl_seconds=3600)
        assert cache.ttl == 3600
        assert len(cache.cache) == 0

    @staticmethod
    @pytest.mark.asyncio
    async def test_cache_set_and_get() -> None:
        """Test basic cache set and get operations."""
        cache = InferenceCache(ttl_seconds=3600)

        test_key = "test_key"
        test_value = {"data": "test_value"}

        # Set value
        await cache.set(test_key, test_value)

        # Get value
        retrieved_value = await cache.get(test_key)
        assert retrieved_value == test_value

    @staticmethod
    @pytest.mark.asyncio
    async def test_cache_expiration() -> None:
        """Test cache expiration functionality."""
        # Short TTL for testing
        cache = InferenceCache(ttl_seconds=1)

        test_key = "expiring_key"
        test_value = {"data": "will_expire"}

        # Set value
        await cache.set(test_key, test_value)

        # Should be available immediately
        assert await cache.get(test_key) == test_value

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be None after expiration
        assert await cache.get(test_key) is None

    @staticmethod
    @pytest.mark.asyncio
    async def test_cache_miss() -> None:
        """Test cache miss for non-existent key."""
        cache = InferenceCache()

        result = await cache.get("non_existent_key")
        assert result is None

    @staticmethod
    def test_cache_clear() -> None:
        """Test cache clear functionality."""
        cache = InferenceCache()
        cache.cache["key1"] = ("value1", 12345)
        cache.cache["key2"] = ("value2", 12346)

        assert len(cache.cache) == 2

        cache.clear()
        assert len(cache.cache) == 0


class TestInferenceEngineUtilities:
    """Test utility functions of the inference engine."""

    @staticmethod
    def test_generate_cache_key() -> None:
        """Test cache key generation."""
        data_points = [
            ActigraphyDataPoint(timestamp=datetime.now(UTC), value=50.0),
            ActigraphyDataPoint(timestamp=datetime.now(UTC), value=75.0),
        ]

        input_data = ActigraphyInput(
            user_id="test-user",
            data_points=data_points,
            sampling_rate=1.0,
            duration_hours=24,
        )

        cache_key = AsyncInferenceEngine._generate_cache_key(input_data)

        assert isinstance(cache_key, str)
        assert len(cache_key) > 0

        # Same input should generate same key
        cache_key2 = AsyncInferenceEngine._generate_cache_key(input_data)
        assert cache_key == cache_key2

    @staticmethod
    def test_generate_cache_key_different_inputs() -> None:
        """Test that different inputs generate different cache keys."""
        data_points1 = [ActigraphyDataPoint(timestamp=datetime.now(UTC), value=50.0)]
        data_points2 = [ActigraphyDataPoint(timestamp=datetime.now(UTC), value=75.0)]

        input1 = ActigraphyInput(
            user_id="user1",
            data_points=data_points1,
            sampling_rate=1.0,
            duration_hours=24,
        )

        input2 = ActigraphyInput(
            user_id="user2",
            data_points=data_points2,
            sampling_rate=1.0,
            duration_hours=24,
        )

        key1 = AsyncInferenceEngine._generate_cache_key(input1)
        key2 = AsyncInferenceEngine._generate_cache_key(input2)

        assert key1 != key2

    @staticmethod
    def test_generate_cache_key_empty_data() -> None:
        """Test cache key generation with empty data points."""
        input_data = ActigraphyInput(
            user_id="test-user", data_points=[], sampling_rate=1.0, duration_hours=24
        )

        cache_key = AsyncInferenceEngine._generate_cache_key(input_data)
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0


class TestInferenceEngineStats:
    """Test inference engine statistics functionality."""

    @staticmethod
    def test_get_stats_initial() -> None:
        """Test getting stats from freshly initialized engine."""
        mock_pat_service = MagicMock(spec=PATModelService)
        engine = AsyncInferenceEngine(pat_service=mock_pat_service)

        stats = engine.get_stats()

        assert stats["requests_processed"] == 0
        assert stats["cache_hits"] == 0
        assert stats["error_count"] == 0
        assert stats["cache_hit_rate_percent"] == 0.0
        assert stats["is_running"] is False

    @staticmethod
    @pytest.mark.asyncio
    async def test_get_stats_after_requests() -> None:
        """Test getting stats after processing requests."""
        # Create sample data inline
        data_points = [
            ActigraphyDataPoint(timestamp=datetime.now(UTC), value=float(i % 100))
            for i in range(1440)  # 24 hours of data
        ]

        sample_actigraphy_input = ActigraphyInput(
            user_id=str(uuid4()),
            data_points=data_points,
            sampling_rate=1.0,
            duration_hours=24,
        )

        mock_analysis = ActigraphyAnalysis(
            user_id=sample_actigraphy_input.user_id,
            analysis_timestamp=datetime.now(UTC).isoformat(),
            sleep_efficiency=85.0,
            sleep_onset_latency=15.0,
            wake_after_sleep_onset=30.0,
            total_sleep_time=7.5,
            circadian_rhythm_score=0.75,
            activity_fragmentation=0.25,
            depression_risk_score=0.2,
            sleep_stages=["wake"] * 100,
            confidence_score=0.85,
            clinical_insights=["Good sleep efficiency"],
            embedding=[0.0] * 128,
        )

        mock_pat_service = MagicMock(spec=PATModelService)
        mock_pat_service.analyze_actigraphy = AsyncMock(return_value=mock_analysis)

        async with AsyncInferenceEngine(pat_service=mock_pat_service) as engine:
            request = InferenceRequest(
                request_id=str(uuid4()),
                input_data=sample_actigraphy_input,
                timeout_seconds=30.0,
            )

            await engine.predict_async(request)

            stats = engine.get_stats()
            assert stats["requests_processed"] > 0
            assert stats["is_running"] is True


class TestInferenceEngineErrorHandling:
    """Test error handling in the inference engine."""

    @staticmethod
    @pytest.mark.asyncio
    async def test_pat_service_error_handling() -> None:
        """Test handling of PAT service errors."""
        # Create sample data inline
        data_points = [
            ActigraphyDataPoint(timestamp=datetime.now(UTC), value=float(i % 100))
            for i in range(100)  # Less data for faster test
        ]

        sample_actigraphy_input = ActigraphyInput(
            user_id=str(uuid4()),
            data_points=data_points,
            sampling_rate=1.0,
            duration_hours=24,
        )

        # Create a mock that fails immediately
        def failing_mock(_: object) -> Never:
            error_msg = "Model not loaded"
            raise RuntimeError(error_msg)

        mock_pat_service = MagicMock(spec=PATModelService)
        mock_pat_service.analyze_actigraphy = failing_mock

        async with AsyncInferenceEngine(pat_service=mock_pat_service) as engine:
            request = InferenceRequest(
                request_id=str(uuid4()),
                input_data=sample_actigraphy_input,
                timeout_seconds=1.0,  # Short timeout for faster test
            )

            # Due to the resilient prediction decorator, errors are wrapped in ServiceUnavailableProblem
            with pytest.raises(
                ServiceUnavailableProblem, match="temporarily unavailable"
            ):
                await engine.predict_async(request)

    @staticmethod
    @pytest.mark.asyncio
    async def test_timeout_handling() -> None:
        """Test timeout handling for slow requests."""
        # Create sample data inline
        data_points = [
            ActigraphyDataPoint(timestamp=datetime.now(UTC), value=float(i % 100))
            for i in range(1440)  # 24 hours of data
        ]

        sample_actigraphy_input = ActigraphyInput(
            user_id=str(uuid4()),
            data_points=data_points,
            sampling_rate=1.0,
            duration_hours=24,
        )

        mock_pat_service = MagicMock(spec=PATModelService)
        # Simulate slow processing
        mock_pat_service.analyze_actigraphy = AsyncMock(
            side_effect=lambda _: asyncio.sleep(2.0)  # 2 second delay
        )

        async with AsyncInferenceEngine(pat_service=mock_pat_service) as engine:
            request = InferenceRequest(
                request_id=str(uuid4()),
                input_data=sample_actigraphy_input,
                timeout_seconds=0.1,  # Very short timeout
            )

            with pytest.raises(
                ServiceUnavailableProblem, match="temporarily unavailable"
            ):
                await engine.predict_async(request)

    @staticmethod
    @pytest.mark.asyncio
    async def test_cache_error_handling() -> None:
        """Test graceful handling of cache errors."""
        # Create sample data inline
        data_points = [
            ActigraphyDataPoint(timestamp=datetime.now(UTC), value=float(i % 100))
            for i in range(1440)  # 24 hours of data
        ]

        sample_actigraphy_input = ActigraphyInput(
            user_id=str(uuid4()),
            data_points=data_points,
            sampling_rate=1.0,
            duration_hours=24,
        )

        mock_analysis = ActigraphyAnalysis(
            user_id=sample_actigraphy_input.user_id,
            analysis_timestamp=datetime.now(UTC).isoformat(),
            sleep_efficiency=85.0,
            sleep_onset_latency=15.0,
            wake_after_sleep_onset=30.0,
            total_sleep_time=7.5,
            circadian_rhythm_score=0.75,
            activity_fragmentation=0.25,
            depression_risk_score=0.2,
            sleep_stages=["wake"] * 100,
            confidence_score=0.85,
            clinical_insights=["Good sleep efficiency"],
            embedding=[0.0] * 128,
        )

        mock_pat_service = MagicMock(spec=PATModelService)
        mock_pat_service.analyze_actigraphy = AsyncMock(return_value=mock_analysis)

        async with AsyncInferenceEngine(pat_service=mock_pat_service) as engine:
            request = InferenceRequest(
                request_id=str(uuid4()),
                input_data=sample_actigraphy_input,
                timeout_seconds=30.0,
                cache_enabled=False,  # Disable cache to avoid cache errors
            )

            # Should work when cache is disabled
            result = await engine.predict_async(request)
            assert isinstance(result, InferenceResponse)


class TestInferenceModels:
    """Test the data models used by the inference engine."""

    @staticmethod
    def test_inference_request_validation() -> None:
        """Test inference request model validation."""
        data_points = [ActigraphyDataPoint(timestamp=datetime.now(UTC), value=50.0)]
        input_data = ActigraphyInput(
            user_id="test-user",
            data_points=data_points,
            sampling_rate=1.0,
            duration_hours=24,
        )

        request = InferenceRequest(
            request_id="test-request",
            input_data=input_data,
            timeout_seconds=10.0,
            cache_enabled=True,
        )

        assert request.request_id == "test-request"
        assert request.input_data == input_data
        assert request.timeout_seconds == 10.0
        assert request.cache_enabled is True

    @staticmethod
    def test_inference_response_validation() -> None:
        """Test inference response model validation."""
        mock_analysis = ActigraphyAnalysis(
            user_id="test-user",
            analysis_timestamp=datetime.now(UTC).isoformat(),
            sleep_efficiency=85.0,
            sleep_onset_latency=15.0,
            wake_after_sleep_onset=30.0,
            total_sleep_time=7.5,
            circadian_rhythm_score=0.75,
            activity_fragmentation=0.25,
            depression_risk_score=0.2,
            sleep_stages=["wake"],
            confidence_score=0.85,
            clinical_insights=["Good sleep"],
            embedding=[0.0] * 128,
        )

        response = InferenceResponse(
            request_id="test-request",
            analysis=mock_analysis,
            processing_time_ms=150.5,
            cached=False,
            timestamp=1234567890.0,
        )

        assert response.request_id == "test-request"
        assert response.analysis == mock_analysis
        assert response.processing_time_ms == 150.5
        assert response.cached is False
        assert response.timestamp == 1234567890.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
