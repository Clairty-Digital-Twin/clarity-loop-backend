"""Comprehensive unit tests for PAT Predictor.

Tests all components of the PAT predictor including:
- PredictionRequest and PredictionResult dataclasses
- PredictionCache with LRU behavior
- BatchProcessor for efficient batching
- PATPredictor main functionality
- Error handling and edge cases
"""

from __future__ import annotations

from collections import deque
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from clarity.ml.pat_model_loader import ModelSize, PATModelLoader
from clarity.ml.pat_predictor import (
    BatchProcessor,
    PATPredictor,
    PredictionCache,
    PredictionError,
    PredictionRequest,
    PredictionResult,
)


class TestPredictionRequest:
    """Test PredictionRequest dataclass."""

    def test_prediction_request_creation(self) -> None:
        """Test creating a PredictionRequest with all parameters."""
        data = np.random.randn(4, 10080).astype(np.float32)
        request = PredictionRequest(
            data=data,
            model_size=ModelSize.LARGE,
            return_embeddings=False,
            enable_batching=False,
        )

        assert np.array_equal(request.data, data)
        assert request.model_size == ModelSize.LARGE
        assert request.return_embeddings is False
        assert request.enable_batching is False

    def test_prediction_request_defaults(self) -> None:
        """Test PredictionRequest with default values."""
        data = np.random.randn(1, 10080).astype(np.float32)
        request = PredictionRequest(data=data)

        assert request.model_size == ModelSize.MEDIUM
        assert request.return_embeddings is True
        assert request.enable_batching is True


class TestPredictionResult:
    """Test PredictionResult dataclass."""

    def test_prediction_result_creation(self) -> None:
        """Test creating a PredictionResult with all fields."""
        embeddings = np.random.randn(2, 560, 96).astype(np.float32)
        seq_embeddings = np.random.randn(2, 96).astype(np.float32)

        result = PredictionResult(
            embeddings=embeddings,
            sequence_embeddings=seq_embeddings,
            inference_time_ms=15.5,
            model_version="v1.2",
            batch_size=2,
        )

        assert np.array_equal(result.embeddings, embeddings)
        assert np.array_equal(result.sequence_embeddings, seq_embeddings)
        assert result.inference_time_ms == 15.5
        assert result.model_version == "v1.2"
        assert result.batch_size == 2

    def test_prediction_result_defaults(self) -> None:
        """Test PredictionResult with default values."""
        result = PredictionResult()

        assert result.embeddings is None
        assert result.sequence_embeddings is None
        assert result.inference_time_ms == 0.0
        assert result.model_version == ""
        assert result.batch_size == 1


class TestPredictionCache:
    """Test PredictionCache functionality."""

    def test_cache_initialization(self) -> None:
        """Test PredictionCache initialization."""
        cache = PredictionCache(max_size=100)

        assert cache._max_size == 100
        assert len(cache._cache) == 0
        assert isinstance(cache._access_order, deque)

    def test_cache_key_generation(self) -> None:
        """Test cache key generation."""
        cache = PredictionCache()
        data = np.array([[1, 2, 3]], dtype=np.float32)

        key1 = cache._get_key(data, ModelSize.SMALL)
        key2 = cache._get_key(data, ModelSize.SMALL)
        key3 = cache._get_key(data, ModelSize.LARGE)

        # Same data and size should produce same key
        assert key1 == key2
        # Different size should produce different key
        assert key1 != key3

    def test_cache_set_and_get(self) -> None:
        """Test setting and getting from cache."""
        cache = PredictionCache()
        data = np.random.randn(1, 100).astype(np.float32)
        result = PredictionResult(inference_time_ms=10.0)

        cache.set(data, ModelSize.MEDIUM, result)
        retrieved = cache.get(data, ModelSize.MEDIUM)

        assert retrieved is result

    def test_cache_get_nonexistent(self) -> None:
        """Test getting non-existent entry."""
        cache = PredictionCache()
        data = np.random.randn(1, 100).astype(np.float32)

        result = cache.get(data, ModelSize.SMALL)
        assert result is None

    def test_cache_lru_behavior(self) -> None:
        """Test LRU eviction behavior."""
        cache = PredictionCache(max_size=2)

        # Add three items to cache with max_size=2
        data1 = np.array([[1]], dtype=np.float32)
        data2 = np.array([[2]], dtype=np.float32)
        data3 = np.array([[3]], dtype=np.float32)

        result1 = PredictionResult(model_version="v1")
        result2 = PredictionResult(model_version="v2")
        result3 = PredictionResult(model_version="v3")

        cache.set(data1, ModelSize.SMALL, result1)
        cache.set(data2, ModelSize.SMALL, result2)

        # Access data1 to make it more recent
        cache.get(data1, ModelSize.SMALL)

        # Add third item - should evict data2 (least recently used)
        cache.set(data3, ModelSize.SMALL, result3)

        assert cache.get(data1, ModelSize.SMALL) is result1  # Still in cache
        assert cache.get(data2, ModelSize.SMALL) is None  # Evicted
        assert cache.get(data3, ModelSize.SMALL) is result3  # In cache

    def test_cache_update_existing(self) -> None:
        """Test updating existing cache entry."""
        cache = PredictionCache(max_size=2)
        data = np.array([[1]], dtype=np.float32)

        result1 = PredictionResult(model_version="v1")
        result2 = PredictionResult(model_version="v2")

        cache.set(data, ModelSize.SMALL, result1)
        cache.set(data, ModelSize.SMALL, result2)

        # Should have updated the value
        assert cache.get(data, ModelSize.SMALL) is result2
        assert len(cache._cache) == 1  # No duplicate entries


class TestBatchProcessor:
    """Test BatchProcessor functionality."""

    def test_batch_processor_initialization(self) -> None:
        """Test BatchProcessor initialization."""
        processor = BatchProcessor(max_batch_size=16, batch_timeout_ms=100)

        assert processor.max_batch_size == 16
        assert processor.batch_timeout_ms == 100
        assert processor._pending_batch == []
        assert processor._batch_start_time is None

    def test_add_to_batch_below_max_size(self) -> None:
        """Test adding data when batch is not full."""
        processor = BatchProcessor(max_batch_size=3)

        data1 = np.array([1, 2, 3], dtype=np.float32)
        data2 = np.array([4, 5, 6], dtype=np.float32)

        result1 = processor.add_to_batch(data1)
        result2 = processor.add_to_batch(data2)

        assert result1 is None  # Batch not ready
        assert result2 is None  # Still not full
        assert len(processor._pending_batch) == 2

    def test_add_to_batch_reaches_max_size(self) -> None:
        """Test batch is returned when max size is reached."""
        processor = BatchProcessor(max_batch_size=2)

        data1 = np.array([1, 2], dtype=np.float32)
        data2 = np.array([3, 4], dtype=np.float32)

        result1 = processor.add_to_batch(data1)
        result2 = processor.add_to_batch(data2)

        assert result1 is None
        assert result2 is not None
        assert result2.shape == (2, 2)
        assert np.array_equal(result2[0], data1)
        assert np.array_equal(result2[1], data2)

        # Batch should be cleared
        assert len(processor._pending_batch) == 0
        assert processor._batch_start_time is None

    def test_add_to_batch_timeout(self) -> None:
        """Test batch is returned on timeout."""
        processor = BatchProcessor(max_batch_size=10, batch_timeout_ms=50)

        data = np.array([1, 2], dtype=np.float32)
        result1 = processor.add_to_batch(data)
        assert result1 is None

        # Wait for timeout
        time.sleep(0.1)

        data2 = np.array([3, 4], dtype=np.float32)
        result2 = processor.add_to_batch(data2)

        # Should return batch due to timeout (includes both items since data2 was added before timeout check)
        assert result2 is not None
        assert result2.shape == (2, 2)  # Both items
        assert np.array_equal(result2[0], data)
        assert np.array_equal(result2[1], data2)

    def test_get_and_clear_batch_empty(self) -> None:
        """Test getting empty batch."""
        processor = BatchProcessor()
        batch = processor._get_and_clear_batch()

        assert isinstance(batch, np.ndarray)
        assert batch.shape == (0,)


class TestPATPredictor:
    """Test PATPredictor functionality."""

    @pytest.fixture
    def mock_model_loader(self) -> MagicMock:
        """Create mock model loader."""
        loader = MagicMock(spec=PATModelLoader)
        loader._current_versions = {}
        return loader

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create mock PAT model."""
        model = MagicMock(spec=nn.Module)
        # Mock forward pass
        model.return_value = torch.randn(
            1, 560, 96
        )  # batch_size=1, patches=560, embed_dim=96
        return model

    @pytest.fixture
    def predictor(self, mock_model_loader: MagicMock) -> PATPredictor:
        """Create PATPredictor instance."""
        return PATPredictor(
            model_loader=mock_model_loader,
            enable_caching=True,
            cache_size=100,
            enable_batching=True,
            max_batch_size=32,
        )

    def test_predictor_initialization(self, mock_model_loader: MagicMock) -> None:
        """Test PATPredictor initialization."""
        with patch("clarity.ml.pat_predictor.logger") as mock_logger:
            predictor = PATPredictor(
                model_loader=mock_model_loader,
                enable_caching=False,
                cache_size=50,
                enable_batching=False,
                max_batch_size=16,
            )

            assert predictor.model_loader is mock_model_loader
            assert predictor.enable_caching is False
            assert predictor.enable_batching is False
            assert predictor._cache is None
            assert predictor._batch_processor is None

            mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_predict_with_embeddings(
        self,
        predictor: PATPredictor,
        mock_model_loader: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test prediction with embeddings returned."""
        # Setup
        mock_model_loader.load_model = AsyncMock(return_value=mock_model)
        mock_model_loader.get_current_version = MagicMock(
            return_value=MagicMock(version="v1.0")
        )

        data = np.random.randn(1, 10080).astype(np.float32)
        request = PredictionRequest(data=data, return_embeddings=True)

        # Predict
        result = await predictor.predict(request)

        assert result.embeddings is not None
        assert result.sequence_embeddings is not None
        assert result.embeddings.shape == (1, 560, 96)
        assert result.sequence_embeddings.shape == (1, 96)
        assert result.model_version == "v1.0"
        assert result.batch_size == 1
        assert result.inference_time_ms > 0

    @pytest.mark.asyncio
    async def test_predict_sequence_only(
        self,
        predictor: PATPredictor,
        mock_model_loader: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test prediction with only sequence embeddings."""
        # Setup
        mock_model_loader.load_model = AsyncMock(return_value=mock_model)

        data = np.random.randn(1, 10080).astype(np.float32)
        request = PredictionRequest(data=data, return_embeddings=False)

        # Predict
        result = await predictor.predict(request)

        assert result.embeddings is None
        assert result.sequence_embeddings is not None
        assert result.sequence_embeddings.shape == (1, 96)

    @pytest.mark.asyncio
    async def test_predict_with_cache_hit(
        self, predictor: PATPredictor, mock_model_loader: MagicMock
    ) -> None:
        """Test prediction with cache hit."""
        # Add to cache
        data = np.array([[1, 2, 3]], dtype=np.float32)
        cached_result = PredictionResult(
            inference_time_ms=5.0,
            model_version="cached",
        )
        predictor._cache.set(data, ModelSize.MEDIUM, cached_result)

        # Predict (should hit cache)
        request = PredictionRequest(data=data)
        result = await predictor.predict(request)

        assert result is cached_result
        assert predictor._cache_hits == 1
        assert predictor._cache_misses == 0

        # Model loader should not be called
        mock_model_loader.load_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_predict_with_cache_miss(
        self,
        predictor: PATPredictor,
        mock_model_loader: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test prediction with cache miss."""
        mock_model_loader.load_model = AsyncMock(return_value=mock_model)

        data = np.random.randn(1, 10080).astype(np.float32)
        request = PredictionRequest(data=data)

        result = await predictor.predict(request)

        assert predictor._cache_hits == 0
        assert predictor._cache_misses == 1

        # Result should be cached
        cached = predictor._cache.get(data, ModelSize.MEDIUM)
        assert cached is result

    @pytest.mark.asyncio
    async def test_predict_error_handling(
        self, predictor: PATPredictor, mock_model_loader: MagicMock
    ) -> None:
        """Test error handling in prediction."""
        mock_model_loader.load_model = AsyncMock(side_effect=Exception("Model error"))

        data = np.random.randn(1, 10080).astype(np.float32)
        request = PredictionRequest(data=data)

        with pytest.raises(PredictionError, match="Prediction failed"):
            await predictor.predict(request)

    @pytest.mark.asyncio
    async def test_predict_batch(
        self,
        predictor: PATPredictor,
        mock_model_loader: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test batch prediction."""

        # Setup model to handle different batch sizes
        def mock_forward(x):
            batch_size = x.shape[0]
            return torch.randn(batch_size, 560, 96)

        mock_model.side_effect = mock_forward
        mock_model_loader.load_model = AsyncMock(return_value=mock_model)

        # Create multiple requests
        requests = [
            PredictionRequest(
                data=np.random.randn(1, 10080).astype(np.float32),
                model_size=ModelSize.SMALL,
            ),
            PredictionRequest(
                data=np.random.randn(1, 10080).astype(np.float32),
                model_size=ModelSize.SMALL,
            ),
            PredictionRequest(
                data=np.random.randn(1, 10080).astype(np.float32),
                model_size=ModelSize.MEDIUM,
            ),
        ]

        # Predict batch
        results = await predictor.predict_batch(requests)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, PredictionResult)
            assert result.batch_size == 1

    @pytest.mark.asyncio
    async def test_predict_batch_grouped_by_size(
        self,
        predictor: PATPredictor,
        mock_model_loader: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test batch prediction groups by model size."""
        call_count = 0

        async def mock_load(size, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_model

        mock_model_loader.load_model = mock_load
        mock_model.return_value = torch.randn(2, 560, 96)  # batch_size=2

        # Create requests with two different model sizes
        requests = [
            PredictionRequest(
                data=np.random.randn(1, 10080).astype(np.float32),
                model_size=ModelSize.SMALL,
            ),
            PredictionRequest(
                data=np.random.randn(1, 10080).astype(np.float32),
                model_size=ModelSize.LARGE,
            ),
        ]

        _ = await predictor.predict_batch(requests)

        # Should load model twice (once per size)
        assert call_count >= 2  # May be more due to internal calls

    def test_prepare_input_float32(self, predictor: PATPredictor) -> None:
        """Test preparing float32 input."""
        data = np.array([[1, 2, 3]], dtype=np.float32)
        tensor = predictor._prepare_input(data)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert tensor.shape == (1, 3)

    def test_prepare_input_conversion(self, predictor: PATPredictor) -> None:
        """Test preparing input with type conversion."""
        data = np.array([[1, 2, 3]], dtype=np.float64)
        tensor = predictor._prepare_input(data)

        assert tensor.dtype == torch.float32

    def test_prepare_input_add_batch_dim(self, predictor: PATPredictor) -> None:
        """Test adding batch dimension to 1D input."""
        data = np.array([1, 2, 3], dtype=np.float32)
        tensor = predictor._prepare_input(data)

        assert tensor.shape == (1, 3)

    def test_get_metrics_with_data(self, predictor: PATPredictor) -> None:
        """Test getting metrics with prediction data."""
        # Add prediction times
        predictor._prediction_times.extend([0.01, 0.02, 0.015])

        # Add cache stats
        predictor._cache_hits = 10
        predictor._cache_misses = 5

        metrics = predictor.get_metrics()

        assert metrics["cache_enabled"] is True
        assert metrics["batch_enabled"] is True
        assert metrics["average_prediction_time_ms"] == 15.0
        assert metrics["total_predictions"] == 3
        assert metrics["cache_hits"] == 10
        assert metrics["cache_misses"] == 5
        assert metrics["cache_hit_rate"] == 0.6666666666666666

    def test_get_metrics_no_cache(self, mock_model_loader: MagicMock) -> None:
        """Test getting metrics without cache enabled."""
        predictor = PATPredictor(
            model_loader=mock_model_loader,
            enable_caching=False,
        )

        predictor._prediction_times.append(0.01)

        metrics = predictor.get_metrics()

        assert metrics["cache_enabled"] is False
        assert "cache_hits" not in metrics

    def test_get_metrics_empty(self, predictor: PATPredictor) -> None:
        """Test getting metrics with no data."""
        metrics = predictor.get_metrics()

        assert metrics["cache_enabled"] is True
        assert metrics["batch_enabled"] is True
        assert "average_prediction_time_ms" not in metrics
