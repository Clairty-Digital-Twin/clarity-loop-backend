"""PAT Predictor Component - Following SOLID principles.

Handles prediction and inference using PAT models.
Extracted from monolithic PATService for better separation of concerns.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from clarity.ml.pat_model_loader import ModelSize, PATModelLoader

logger = logging.getLogger(__name__)


@dataclass
class PredictionRequest:
    """Request for PAT prediction."""
    data: np.ndarray  # Shape: (batch_size, sequence_length)
    model_size: ModelSize = ModelSize.MEDIUM
    return_embeddings: bool = True
    enable_batching: bool = True


@dataclass 
class PredictionResult:
    """Result from PAT prediction."""
    embeddings: Optional[np.ndarray] = None  # Shape: (batch_size, num_patches, embed_dim)
    sequence_embeddings: Optional[np.ndarray] = None  # Shape: (batch_size, embed_dim)
    inference_time_ms: float = 0.0
    model_version: str = ""
    batch_size: int = 1


class PredictionCache:
    """Simple LRU cache for predictions.
    
    Follows Single Responsibility: Only caches predictions.
    """
    
    def __init__(self, max_size: int = 1000):
        """Initialize cache with max size."""
        self._cache: dict[str, PredictionResult] = {}
        self._access_order: deque[str] = deque(maxlen=max_size)
        self._max_size = max_size
    
    def _get_key(self, data: np.ndarray, model_size: ModelSize) -> str:
        """Generate cache key from input data."""
        # Use hash of data for key
        data_hash = hash(data.tobytes())
        return f"{model_size.value}:{data_hash}"
    
    def get(self, data: np.ndarray, model_size: ModelSize) -> Optional[PredictionResult]:
        """Get cached prediction if available."""
        key = self._get_key(data, model_size)
        
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        
        return None
    
    def set(self, data: np.ndarray, model_size: ModelSize, result: PredictionResult) -> None:
        """Store prediction in cache."""
        key = self._get_key(data, model_size)
        
        # Remove oldest if at capacity
        if len(self._cache) >= self._max_size and key not in self._cache:
            oldest = self._access_order.popleft()
            del self._cache[oldest]
        
        self._cache[key] = result
        self._access_order.append(key)


class BatchProcessor:
    """Handles batching of predictions for efficiency.
    
    Follows Single Responsibility: Only handles batching logic.
    """
    
    def __init__(self, max_batch_size: int = 32, batch_timeout_ms: int = 50):
        """Initialize batch processor.
        
        Args:
            max_batch_size: Maximum batch size
            batch_timeout_ms: Maximum time to wait for batch to fill
        """
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self._pending_batch: list[np.ndarray] = []
        self._batch_start_time: Optional[float] = None
    
    def add_to_batch(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Add data to batch, return full batch if ready.
        
        Args:
            data: Input data to add to batch
            
        Returns:
            Full batch if ready, None otherwise
        """
        if self._batch_start_time is None:
            self._batch_start_time = time.time()
        
        self._pending_batch.append(data)
        
        # Check if batch is ready
        if len(self._pending_batch) >= self.max_batch_size:
            return self._get_and_clear_batch()
        
        # Check timeout
        elapsed_ms = (time.time() - self._batch_start_time) * 1000
        if elapsed_ms >= self.batch_timeout_ms:
            return self._get_and_clear_batch()
        
        return None
    
    def _get_and_clear_batch(self) -> np.ndarray:
        """Get current batch and clear state."""
        if not self._pending_batch:
            return np.array([])
        
        batch = np.stack(self._pending_batch)
        self._pending_batch = []
        self._batch_start_time = None
        
        return batch


class PATPredictor:
    """Handles PAT model predictions with optimization.
    
    Follows SOLID principles:
    - Single Responsibility: Only handles predictions
    - Open/Closed: Extensible for new prediction types
    - Dependency Inversion: Depends on ModelLoader abstraction
    """
    
    def __init__(
        self,
        model_loader: PATModelLoader,
        enable_caching: bool = True,
        cache_size: int = 1000,
        enable_batching: bool = True,
        max_batch_size: int = 32,
    ):
        """Initialize predictor.
        
        Args:
            model_loader: Model loader instance
            enable_caching: Enable prediction caching
            cache_size: Maximum cache size
            enable_batching: Enable request batching
            max_batch_size: Maximum batch size
        """
        self.model_loader = model_loader
        self.enable_caching = enable_caching
        self.enable_batching = enable_batching
        
        # Initialize cache
        self._cache = PredictionCache(cache_size) if enable_caching else None
        
        # Initialize batch processor
        self._batch_processor = BatchProcessor(max_batch_size) if enable_batching else None
        
        # Metrics
        self._prediction_times: deque[float] = deque(maxlen=1000)
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(
            "PATPredictor initialized with caching=%s, batching=%s",
            enable_caching, enable_batching
        )
    
    async def predict(self, request: PredictionRequest) -> PredictionResult:
        """Make prediction using PAT model.
        
        Args:
            request: Prediction request
            
        Returns:
            Prediction result
        """
        start_time = time.time()
        
        try:
            # Check cache first
            if self.enable_caching and self._cache:
                cached = self._cache.get(request.data, request.model_size)
                if cached:
                    self._cache_hits += 1
                    logger.debug("Prediction cache hit")
                    return cached
                self._cache_misses += 1
            
            # Load model
            model = await self.model_loader.load_model(request.model_size)
            
            # Get current version
            version_info = self.model_loader._current_versions.get(request.model_size)
            model_version = version_info.version if version_info else "unknown"
            
            # Prepare data
            input_tensor = self._prepare_input(request.data)
            
            # Make prediction
            with torch.no_grad():
                if request.return_embeddings:
                    embeddings = model.get_patch_embeddings(input_tensor)
                    sequence_embeddings = embeddings.mean(dim=1)  # Mean pool
                    
                    result = PredictionResult(
                        embeddings=embeddings.cpu().numpy(),
                        sequence_embeddings=sequence_embeddings.cpu().numpy(),
                        inference_time_ms=(time.time() - start_time) * 1000,
                        model_version=model_version,
                        batch_size=input_tensor.shape[0]
                    )
                else:
                    # Just sequence embeddings
                    sequence_embeddings = model.get_sequence_embedding(input_tensor)
                    
                    result = PredictionResult(
                        sequence_embeddings=sequence_embeddings.cpu().numpy(),
                        inference_time_ms=(time.time() - start_time) * 1000,
                        model_version=model_version,
                        batch_size=input_tensor.shape[0]
                    )
            
            # Cache result
            if self.enable_caching and self._cache:
                self._cache.set(request.data, request.model_size, result)
            
            # Record metrics
            self._prediction_times.append(time.time() - start_time)
            
            return result
            
        except Exception as e:
            logger.exception("Prediction failed")
            raise PredictionError(f"Prediction failed: {e}") from e
    
    async def predict_batch(
        self, 
        requests: list[PredictionRequest]
    ) -> list[PredictionResult]:
        """Make predictions for multiple requests.
        
        Optimizes by batching similar requests together.
        
        Args:
            requests: List of prediction requests
            
        Returns:
            List of prediction results
        """
        # Group by model size
        grouped: dict[ModelSize, list[tuple[int, PredictionRequest]]] = {}
        for idx, req in enumerate(requests):
            if req.model_size not in grouped:
                grouped[req.model_size] = []
            grouped[req.model_size].append((idx, req))
        
        # Process each group
        results: list[Optional[PredictionResult]] = [None] * len(requests)
        
        for model_size, group in grouped.items():
            # Extract data and indices
            indices = [idx for idx, _ in group]
            data_list = [req.data for _, req in group]
            
            # Stack into batch
            batch_data = np.stack(data_list)
            
            # Create batch request
            batch_request = PredictionRequest(
                data=batch_data,
                model_size=model_size,
                return_embeddings=group[0][1].return_embeddings,
                enable_batching=False  # Already batched
            )
            
            # Make prediction
            batch_result = await self.predict(batch_request)
            
            # Split results
            for i, idx in enumerate(indices):
                if batch_result.embeddings is not None:
                    embeddings = batch_result.embeddings[i:i+1]
                else:
                    embeddings = None
                
                if batch_result.sequence_embeddings is not None:
                    seq_embeddings = batch_result.sequence_embeddings[i:i+1]
                else:
                    seq_embeddings = None
                
                results[idx] = PredictionResult(
                    embeddings=embeddings,
                    sequence_embeddings=seq_embeddings,
                    inference_time_ms=batch_result.inference_time_ms,
                    model_version=batch_result.model_version,
                    batch_size=1
                )
        
        return [r for r in results if r is not None]
    
    def _prepare_input(self, data: np.ndarray) -> torch.Tensor:
        """Prepare input data for model.
        
        Args:
            data: Input numpy array
            
        Returns:
            Prepared torch tensor
        """
        # Ensure float32
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # Convert to tensor
        tensor = torch.from_numpy(data)
        
        # Add batch dimension if needed
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def get_metrics(self) -> dict[str, Any]:
        """Get prediction metrics."""
        metrics = {
            "cache_enabled": self.enable_caching,
            "batch_enabled": self.enable_batching,
        }
        
        if self._prediction_times:
            metrics.update({
                "average_prediction_time_ms": (
                    sum(self._prediction_times) / len(self._prediction_times) * 1000
                ),
                "total_predictions": len(self._prediction_times),
            })
        
        if self.enable_caching:
            total_cache_requests = self._cache_hits + self._cache_misses
            cache_hit_rate = (
                self._cache_hits / total_cache_requests 
                if total_cache_requests > 0 else 0
            )
            metrics.update({
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_rate": cache_hit_rate,
            })
        
        return metrics


class PredictionError(Exception):
    """Error during prediction."""
    pass