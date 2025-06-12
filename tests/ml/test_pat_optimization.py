"""Comprehensive tests for PAT optimization functionality.

Tests cover:
- Model optimization with TorchScript and pruning
- Performance caching and warmup
- Batch processing
- Error handling and edge cases
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import time
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
import torch

from clarity.ml.pat_optimization import (
    BatchAnalysisProcessor,
    PATPerformanceOptimizer,
    initialize_pat_optimizer,
)
from clarity.ml.pat_service import ActigraphyAnalysis, ActigraphyInput, PATModelService
from clarity.ml.preprocessing import ActigraphyDataPoint

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def mock_pat_service() -> Mock:
    """Mock PAT model service."""
    service = Mock(spec=PATModelService)
    service.is_loaded = True
    service.device = torch.device("cpu")
    service.model = Mock()
    # Set up AsyncMock properly for analyze_actigraphy
    analyze_mock = AsyncMock()
    service.analyze_actigraphy = analyze_mock
    # Set up regular Mocks for internal methods
    preprocess_mock = Mock()
    postprocess_mock = Mock()
    service._preprocess_actigraphy_data = preprocess_mock
    service._postprocess_predictions = postprocess_mock
    return service


@pytest.fixture
def optimizer(mock_pat_service: Mock) -> PATPerformanceOptimizer:
    """Create PAT performance optimizer."""
    return PATPerformanceOptimizer(mock_pat_service)


@pytest.fixture
def sample_actigraphy_data() -> list[ActigraphyDataPoint]:
    """Sample actigraphy data for testing."""
    base_time = datetime.now(UTC)
    return [
        ActigraphyDataPoint(
            timestamp=base_time + timedelta(minutes=i),
            value=30.0 + (10.0 if i % 2 else 0.0),  # Alternating pattern
        )
        for i in range(100)  # 100 data points
    ]


@pytest.fixture
def sample_actigraphy_input(
    sample_actigraphy_data: list[ActigraphyDataPoint],
) -> ActigraphyInput:
    """Sample actigraphy input for testing."""
    return ActigraphyInput(
        user_id=str(uuid4()),
        data_points=sample_actigraphy_data,
    )


@pytest.fixture
def sample_analysis_result() -> ActigraphyAnalysis:
    """Sample analysis result for testing."""
    return ActigraphyAnalysis(
        user_id=str(uuid4()),
        analysis_timestamp=datetime.now(UTC).isoformat(),
        sleep_stages=["wake", "light", "deep", "rem", "wake"]
        * 20,  # 100 stages as strings
        confidence_score=0.85,  # Overall confidence
        sleep_efficiency=92.0,
        sleep_onset_latency=15.5,
        wake_after_sleep_onset=45.2,
        total_sleep_time=7.0,  # hours
        circadian_rhythm_score=0.78,
        activity_fragmentation=0.23,
        depression_risk_score=0.15,
        clinical_insights=[
            "Good sleep efficiency detected",
            "Normal circadian rhythm pattern",
        ],
        embedding=[0.0] * 128,
    )


class TestPATPerformanceOptimizer:  # noqa: PLR0904
    """Test PAT performance optimizer."""

    @staticmethod
    def test_initialization(mock_pat_service: Mock) -> None:
        """Test optimizer initialization."""
        optimizer = PATPerformanceOptimizer(mock_pat_service)

        assert optimizer.pat_service is mock_pat_service
        assert optimizer.compiled_model is None
        assert optimizer.optimization_enabled is False
        assert optimizer._cache == {}

    @staticmethod
    async def test_optimize_model_success(optimizer: PATPerformanceOptimizer) -> None:
        """Test successful model optimization."""
        with (
            patch.object(optimizer, "_apply_model_pruning") as mock_prune,
            patch.object(
                optimizer, "_compile_torchscript", return_value=Mock()
            ) as mock_compile,
        ):
            result = await optimizer.optimize_model(
                use_torchscript=True,
                use_pruning=True,
                pruning_amount=0.1,
            )

            assert result is True
            assert optimizer.optimization_enabled is True
            mock_prune.assert_called_once()
            mock_compile.assert_called_once()

    @staticmethod
    async def test_optimize_model_service_not_loaded(
        optimizer: PATPerformanceOptimizer,
    ) -> None:
        """Test optimization when service not loaded."""
        optimizer.pat_service.is_loaded = False

        result = await optimizer.optimize_model()

        assert result is False
        assert optimizer.optimization_enabled is False

    @staticmethod
    async def test_optimize_model_no_model(optimizer: PATPerformanceOptimizer) -> None:
        """Test optimization when model is None."""
        optimizer.pat_service.model = None

        result = await optimizer.optimize_model()

        assert result is False

    @staticmethod
    async def test_optimize_model_exception(optimizer: PATPerformanceOptimizer) -> None:
        """Test optimization with exception."""
        with patch.object(
            optimizer, "_compile_torchscript", side_effect=Exception("Compile failed")
        ):
            result = await optimizer.optimize_model(use_torchscript=True)

            assert result is False

    @staticmethod
    def test_apply_model_pruning(optimizer: PATPerformanceOptimizer) -> None:
        """Test model pruning application."""
        # Create a simple mock model with linear layers
        model = Mock()
        linear_modules = [
            ("attention.weight", Mock(spec=torch.nn.Linear)),
            ("ff1.weight", Mock(spec=torch.nn.Linear)),
            ("other.weight", Mock(spec=torch.nn.Linear)),
        ]
        model.named_modules.return_value = linear_modules

        with patch("clarity.ml.pat_optimization.prune") as mock_prune:
            optimizer._apply_model_pruning(model, 0.1)

            # Should call pruning on attention and ff modules
            assert mock_prune.l1_unstructured.call_count >= 2

    @staticmethod
    def test_compile_torchscript_success(optimizer: PATPerformanceOptimizer) -> None:
        """Test successful TorchScript compilation."""
        model = Mock()
        model.eval = Mock()

        with (
            patch("torch.jit.trace") as mock_trace,
            patch("torch.jit.optimize_for_inference") as mock_optimize,
        ):
            traced_model = Mock()
            mock_trace.return_value = traced_model
            mock_optimize.return_value = traced_model

            result = optimizer._compile_torchscript(model)

            assert result is traced_model
            model.eval.assert_called_once()
            mock_trace.assert_called_once()

    @staticmethod
    def test_compile_torchscript_no_optimize_for_inference(
        optimizer: PATPerformanceOptimizer,
    ) -> None:
        """Test TorchScript compilation without optimize_for_inference."""
        model = Mock()
        model.eval = Mock()

        with (
            patch("torch.jit.trace") as mock_trace,
            patch("builtins.hasattr", return_value=False),
        ):
            traced_model = Mock()
            mock_trace.return_value = traced_model

            result = optimizer._compile_torchscript(model)

            assert result is traced_model

    @staticmethod
    def test_compile_torchscript_failure(optimizer: PATPerformanceOptimizer) -> None:
        """Test TorchScript compilation failure."""
        model = Mock()
        model.eval = Mock()

        with patch("torch.jit.trace", side_effect=Exception("Trace failed")):
            result = optimizer._compile_torchscript(model)

            assert result is None

    @staticmethod
    def test_save_compiled_model(
        optimizer: PATPerformanceOptimizer, tmp_path: Path
    ) -> None:
        """Test saving compiled model."""
        compiled_model = Mock()
        optimizer.compiled_model = compiled_model
        model_path = tmp_path / "model.pt"

        with patch("torch.jit.save") as mock_save:
            optimizer.save_compiled_model(model_path)

            mock_save.assert_called_once_with(compiled_model, str(model_path))

    @staticmethod
    def test_save_compiled_model_no_model(
        optimizer: PATPerformanceOptimizer, tmp_path: Path
    ) -> None:
        """Test saving when no compiled model exists."""
        model_path = tmp_path / "model.pt"

        optimizer.save_compiled_model(model_path)
        # Should not raise, just log error

    @staticmethod
    def test_generate_cache_key(sample_actigraphy_input: ActigraphyInput) -> None:
        """Test cache key generation."""
        key = PATPerformanceOptimizer._generate_cache_key(sample_actigraphy_input)

        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hex digest length

    @staticmethod
    def test_is_cache_valid() -> None:
        """Test cache validity check."""
        # Valid cache (recent)
        recent_timestamp = time.time() - 1800  # 30 minutes ago
        assert PATPerformanceOptimizer._is_cache_valid(recent_timestamp) is True

        # Invalid cache (old)
        old_timestamp = time.time() - 7200  # 2 hours ago
        assert PATPerformanceOptimizer._is_cache_valid(old_timestamp) is False

    @staticmethod
    async def test_optimized_analyze_cache_hit(
        optimizer: PATPerformanceOptimizer,
        sample_actigraphy_input: ActigraphyInput,
        sample_analysis_result: ActigraphyAnalysis,
    ) -> None:
        """Test optimized analysis with cache hit."""
        # Set up cache
        cache_key = optimizer._generate_cache_key(sample_actigraphy_input)
        optimizer._cache[cache_key] = (sample_analysis_result, time.time())

        result, was_cached = await optimizer.optimized_analyze(sample_actigraphy_input)

        assert result is sample_analysis_result
        assert was_cached is True

    @staticmethod
    async def test_optimized_analyze_cache_miss(
        optimizer: PATPerformanceOptimizer,
        sample_actigraphy_input: ActigraphyInput,
        sample_analysis_result: ActigraphyAnalysis,
    ) -> None:
        """Test optimized analysis with cache miss."""
        # Set up the AsyncMock to return the sample result
        async_mock = AsyncMock(return_value=sample_analysis_result)
        optimizer.pat_service.analyze_actigraphy = async_mock  # type: ignore[method-assign]

        result, was_cached = await optimizer.optimized_analyze(sample_actigraphy_input)

        assert result is sample_analysis_result
        assert was_cached is False
        async_mock.assert_called_once()

    @staticmethod
    async def test_optimized_analyze_with_compiled_model(
        optimizer: PATPerformanceOptimizer,
        sample_actigraphy_input: ActigraphyInput,
        sample_analysis_result: ActigraphyAnalysis,
    ) -> None:
        """Test optimized analysis with compiled model."""
        optimizer.optimization_enabled = True
        optimizer.compiled_model = Mock()

        with patch.object(
            optimizer, "_optimized_inference", return_value=sample_analysis_result
        ):
            result, was_cached = await optimizer.optimized_analyze(
                sample_actigraphy_input
            )

            assert result is sample_analysis_result
            assert was_cached is False

    @staticmethod
    async def test_optimized_inference(
        optimizer: PATPerformanceOptimizer,
        sample_actigraphy_input: ActigraphyInput,
        sample_analysis_result: ActigraphyAnalysis,
    ) -> None:
        """Test optimized inference with compiled model."""
        compiled_model = Mock()
        compiled_model.return_value = torch.randn(1, 100, 5)  # Mock output
        optimizer.compiled_model = compiled_model

        # Mock preprocessing and postprocessing
        input_tensor = torch.randn(100, 1)
        preprocess_mock = Mock(return_value=input_tensor)
        postprocess_mock = Mock(return_value=sample_analysis_result)
        optimizer.pat_service._preprocess_actigraphy_data = preprocess_mock  # type: ignore[method-assign]
        optimizer.pat_service._postprocess_predictions = postprocess_mock  # type: ignore[method-assign]

        result = await optimizer._optimized_inference(sample_actigraphy_input)

        assert result is sample_analysis_result
        preprocess_mock.assert_called_once()
        postprocess_mock.assert_called_once()

    @staticmethod
    async def test_optimized_inference_no_model(
        optimizer: PATPerformanceOptimizer, sample_actigraphy_input: ActigraphyInput
    ) -> None:
        """Test optimized inference without compiled model."""
        optimizer.compiled_model = None

        with pytest.raises(RuntimeError, match="No compiled model available"):
            await optimizer._optimized_inference(sample_actigraphy_input)

    @staticmethod
    def test_clear_cache(optimizer: PATPerformanceOptimizer) -> None:
        """Test cache clearing."""
        mock_analysis = Mock(spec=ActigraphyAnalysis)
        optimizer._cache["test"] = (mock_analysis, 123.0)

        optimizer.clear_cache()

        assert optimizer._cache == {}

    @staticmethod
    def test_get_cache_stats(optimizer: PATPerformanceOptimizer) -> None:
        """Test cache statistics."""
        mock_analysis = Mock(spec=ActigraphyAnalysis)
        optimizer._cache["test1"] = (mock_analysis, time.time())
        optimizer._cache["test2"] = (mock_analysis, time.time() - 100)

        stats = optimizer.get_cache_stats()

        assert stats["cache_size"] == 2
        assert "hit_ratio" in stats
        assert "oldest_entry_age" in stats

    @staticmethod
    async def test_warm_up(optimizer: PATPerformanceOptimizer) -> None:
        """Test model warmup."""
        with patch.object(optimizer, "optimized_analyze") as mock_analyze:
            mock_analyze.return_value = (Mock(), False)

            stats = await optimizer.warm_up(num_iterations=3)

            assert mock_analyze.call_count == 3
            assert "mean_time" in stats
            assert "min_time" in stats
            assert "max_time" in stats


class TestBatchAnalysisProcessor:
    """Test batch analysis processor."""

    @pytest.fixture
    @staticmethod
    def batch_processor(optimizer: PATPerformanceOptimizer) -> BatchAnalysisProcessor:
        """Create batch analysis processor."""
        return BatchAnalysisProcessor(optimizer, max_batch_size=3)

    @staticmethod
    def test_initialization(optimizer: PATPerformanceOptimizer) -> None:
        """Test batch processor initialization."""
        processor = BatchAnalysisProcessor(optimizer, max_batch_size=5)

        assert processor.optimizer is optimizer
        assert processor.max_batch_size == 5
        assert processor.pending_requests == []
        assert processor.processing is False

    @staticmethod
    async def test_analyze_batch_single_request(
        batch_processor: BatchAnalysisProcessor,
        sample_actigraphy_input: ActigraphyInput,
        sample_analysis_result: ActigraphyAnalysis,
    ) -> None:
        """Test batch analysis with single request."""
        batch_processor.optimizer.optimized_analyze = AsyncMock(  # type: ignore[method-assign]
            return_value=(sample_analysis_result, False)
        )

        result = await batch_processor.analyze_batch(sample_actigraphy_input)

        assert result is sample_analysis_result

    @staticmethod
    async def test_analyze_batch_multiple_requests(
        batch_processor: BatchAnalysisProcessor,
        sample_actigraphy_input: ActigraphyInput,
        sample_analysis_result: ActigraphyAnalysis,
    ) -> None:
        """Test batch analysis with multiple requests."""
        batch_processor.optimizer.optimized_analyze = AsyncMock(  # type: ignore[method-assign]
            return_value=(sample_analysis_result, False)
        )

        # Start multiple requests concurrently
        tasks = [
            batch_processor.analyze_batch(sample_actigraphy_input) for _ in range(5)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(result is sample_analysis_result for result in results)

    @staticmethod
    async def test_process_batch_with_exception(
        batch_processor: BatchAnalysisProcessor,
        sample_actigraphy_input: ActigraphyInput,
    ) -> None:
        """Test batch processing with exception."""
        batch_processor.optimizer.optimized_analyze = AsyncMock(  # type: ignore[method-assign]
            side_effect=Exception("Analysis failed")
        )

        with pytest.raises(Exception, match="Analysis failed"):
            await batch_processor.analyze_batch(sample_actigraphy_input)


class TestModuleFunctions:
    """Test module-level functions."""

    @staticmethod
    def test_get_pat_optimizer_not_implemented() -> None:
        """Test get_pat_optimizer raises NotImplementedError."""
        from clarity.ml.pat_optimization import get_pat_optimizer  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="Call initialize_pat_optimizer"):
            get_pat_optimizer()

    @staticmethod
    async def test_initialize_pat_optimizer() -> None:
        """Test PAT optimizer initialization."""
        mock_service = Mock(spec=PATModelService)
        mock_service.is_loaded = False
        mock_service.load_model = AsyncMock()

        # Patch the get_pat_service import inside the initialize function
        with (
            patch("clarity.ml.pat_service.get_pat_service", return_value=mock_service),
            patch.object(
                PATPerformanceOptimizer, "optimize_model", return_value=True
            ) as mock_optimize,
        ):
            optimizer = await initialize_pat_optimizer()

            assert isinstance(optimizer, PATPerformanceOptimizer)
            mock_service.load_model.assert_called_once()
            mock_optimize.assert_called_once()


class TestErrorHandling:
    """Test error handling scenarios."""

    @staticmethod
    async def test_optimization_with_invalid_model(
        optimizer: PATPerformanceOptimizer,
    ) -> None:
        """Test optimization with invalid model structure."""
        # Set up invalid model by setting it to None
        optimizer.pat_service.model = None

        result = await optimizer.optimize_model()

        assert result is False

    @staticmethod
    async def test_cache_corruption_handling(
        optimizer: PATPerformanceOptimizer, sample_actigraphy_input: ActigraphyInput
    ) -> None:
        """Test handling of corrupted cache entries."""
        # Create a corrupted cache entry with wrong format
        cache_key = optimizer._generate_cache_key(sample_actigraphy_input)
        # Simulate cache corruption by setting invalid data
        optimizer._cache[cache_key] = "corrupted_data_not_tuple"  # type: ignore[assignment]

        # Should handle gracefully and not use corrupted cache
        mock_result = Mock(spec=ActigraphyAnalysis)
        async_mock = AsyncMock(return_value=mock_result)
        optimizer.pat_service.analyze_actigraphy = async_mock  # type: ignore[method-assign]

        # Capture logs to verify corruption was detected
        with patch("clarity.ml.pat_optimization.logger") as mock_logger:
            result, was_cached = await optimizer.optimized_analyze(
                sample_actigraphy_input
            )

            # Verify corruption was detected and logged
            assert any(
                "Cache corruption detected" in str(call)
                for call in mock_logger.warning.call_args_list
            )

        assert was_cached is False  # Should not use corrupted cache
        assert result is mock_result
        # After analysis, the cache should contain a new valid entry (not the corrupted one)
        assert cache_key in optimizer._cache  # New valid entry was cached
        cached_result, _timestamp = optimizer._cache[cache_key]
        assert cached_result is mock_result  # Cached the correct result
