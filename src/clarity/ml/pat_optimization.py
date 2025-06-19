"""Performance optimization utilities for PAT model service.

This module provides optimization features including:
- TorchScript compilation for faster inference
- Model pruning for reduced memory usage
- Result caching for repeated analyses
- Batch processing for multiple requests
"""

# removed - breaks FastAPI

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
import hashlib
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

import torch
from torch.nn.utils import prune

from clarity.ml.pat_service import ActigraphyAnalysis, ActigraphyInput, PATModelService
from clarity.ml.preprocessing import ActigraphyDataPoint

if TYPE_CHECKING:
    pass  # Only for type stubs now

logger = logging.getLogger(__name__)

# Constants
CACHE_EXPIRY_HOURS = 1
DEFAULT_WARMUP_ITERATIONS = 5
MAX_BATCH_SIZE = 8
HASH_TRUNCATE_LENGTH = 8


class PATPerformanceOptimizer:
    """Performance optimizer for PAT model service."""

    def __init__(self, pat_service: PATModelService) -> None:
        self.pat_service = pat_service
        self.compiled_model: torch.jit.ScriptModule | None = None
        self.optimization_enabled = False
        self._cache: dict[str, tuple[ActigraphyAnalysis, float]] = {}

    async def optimize_model(
        self,
        *,
        use_torchscript: bool = True,
        use_pruning: bool = False,
        pruning_amount: float = 0.1,
    ) -> bool:
        """Optimize the PAT model for inference performance.

        Args:
            use_torchscript: Enable TorchScript compilation
            use_pruning: Enable model pruning
            pruning_amount: Amount of weights to prune (0.0-1.0)

        Returns:
            True if optimization succeeded
        """
        if not self.pat_service.is_loaded:
            logger.error("Cannot optimize: PAT model not loaded")
            return False

        try:
            model = self.pat_service.model
            if model is None:
                logger.error("Model is None, cannot optimize")
                return False

            # Apply pruning if requested
            if use_pruning:
                logger.info("Applying structured pruning (amount: %s)", pruning_amount)
                self._apply_model_pruning(model, pruning_amount)

            # Compile with TorchScript if requested
            if use_torchscript:
                logger.info("Compiling model with TorchScript")
                self.compiled_model = self._compile_torchscript(model)

                if self.compiled_model is None:
                    logger.warning(
                        "TorchScript compilation failed, using regular model"
                    )
                else:
                    logger.info("TorchScript compilation successful")

            self.optimization_enabled = True
            logger.info("PAT model optimization completed")
        except Exception:
            logger.exception("Model optimization failed")
            return False
        else:
            return True

    @staticmethod
    def _apply_model_pruning(model: torch.nn.Module, amount: float) -> None:
        """Apply structured pruning to reduce model size."""
        # Prune attention layers
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and "attention" in name:
                prune.l1_unstructured(module, name="weight", amount=amount)  # type: ignore[no-untyped-call]

        # Prune feed-forward layers
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and (
                "ff" in name or "feed_forward" in name
            ):
                prune.l1_unstructured(module, name="weight", amount=amount * 0.5)  # type: ignore[no-untyped-call]

        # Remove pruning masks to make pruning permanent
        for _name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                with contextlib.suppress(ValueError):
                    prune.remove(module, "weight")  # type: ignore[no-untyped-call]

    def _compile_torchscript(
        self, model: torch.nn.Module
    ) -> torch.jit.ScriptModule | None:
        """Compile model to TorchScript for optimized inference."""
        try:
            model.eval()

            # Create sample input for tracing
            sample_input = torch.randn(1, 10080, device=self.pat_service.device)

            # Trace the model
            traced_model = torch.jit.trace(model, sample_input)  # type: ignore[no-untyped-call]

            # Optimize for inference
            if hasattr(torch.jit, "optimize_for_inference"):
                return torch.jit.optimize_for_inference(traced_model)
            return traced_model  # type: ignore[no-any-return]

        except Exception:
            logger.exception("TorchScript compilation failed")
            return None

    def save_compiled_model(self, model_path: str | Path) -> None:
        """Save the compiled TorchScript model to disk."""
        if self.compiled_model is None:
            logger.error("No compiled model to save")
            return

        try:
            torch.jit.save(self.compiled_model, str(model_path))  # type: ignore[no-untyped-call]
            logger.info("Compiled model saved to %s", model_path)

        except Exception:
            logger.exception("Failed to save compiled model")

    @staticmethod
    def _generate_cache_key(input_data: ActigraphyInput) -> str:
        """Generate a cache key for actigraphy input."""
        # Create hash from user_id, data points, and parameters
        data_str = f"{input_data.user_id}_{len(input_data.data_points)}"

        # Add sample of data values for uniqueness
        if input_data.data_points:
            first_vals = [dp.value for dp in input_data.data_points[:5]]
            last_vals = [dp.value for dp in input_data.data_points[-5:]]
            data_str += f"_{hash(tuple(first_vals + last_vals))}"

        return hashlib.sha256(data_str.encode()).hexdigest()

    @staticmethod
    def _is_cache_valid(timestamp: float) -> bool:
        """Check if cached result is still valid."""
        return (time.time() - timestamp) < (CACHE_EXPIRY_HOURS * 3600)

    async def optimized_analyze(
        self, input_data: ActigraphyInput, *, use_cache: bool = True
    ) -> tuple[ActigraphyAnalysis, bool]:
        """Optimized analysis with caching and model optimization.

        Returns:
            Tuple of (analysis_result, was_cached)
        """
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(input_data)
            if cache_key in self._cache:
                try:
                    cached_result, timestamp = self._cache[cache_key]
                    if self._is_cache_valid(timestamp):
                        logger.info(
                            "Cache hit for analysis %s",
                            cache_key[:HASH_TRUNCATE_LENGTH],
                        )
                        return cached_result, True
                    # Remove expired entry
                    del self._cache[cache_key]
                except (ValueError, TypeError, AttributeError) as e:
                    # Handle corrupted cache entries gracefully
                    logger.warning(
                        "Cache corruption detected for key %s: %s",
                        cache_key[:HASH_TRUNCATE_LENGTH],
                        e,
                    )
                    # Remove corrupted entry
                    self._cache.pop(cache_key, None)

        # Perform analysis
        start_time = time.time()

        if self.optimization_enabled and self.compiled_model is not None:
            result = await self._optimized_inference(input_data)
        else:
            result = await self.pat_service.analyze_actigraphy(input_data)

        inference_time = time.time() - start_time
        logger.info("Inference completed in %.3fs", inference_time)

        # Cache the result
        if use_cache:
            cache_key = self._generate_cache_key(input_data)
            self._cache[cache_key] = (result, time.time())
            logger.info("Cached analysis result %s", cache_key[:HASH_TRUNCATE_LENGTH])

        return result, False

    async def _optimized_inference(
        self, input_data: ActigraphyInput
    ) -> ActigraphyAnalysis:
        """Run inference using the optimized compiled model."""
        if self.compiled_model is None:
            msg = "No compiled model available"
            raise RuntimeError(msg)

        # Preprocess input data
        input_tensor = self.pat_service._preprocess_actigraphy_data(  # noqa: SLF001
            input_data.data_points
        )
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

        # Run optimized inference
        with torch.no_grad():
            outputs = self.compiled_model(input_tensor)

        # Convert outputs to dictionary format expected by postprocessing
        outputs_dict = {
            "sleep_stage_predictions": (
                outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            ),
            "confidence_scores": torch.softmax(
                outputs[0] if isinstance(outputs, (list, tuple)) else outputs, dim=-1
            ),
            "attention_weights": None,  # Not available in compiled model
            "intermediate_features": None,  # Not available in compiled model
            "model_metadata": {
                "model_type": "compiled_pat",
                "optimization_enabled": True,
                "torchscript_used": True,
            },
        }

        # Post-process results
        return self.pat_service._postprocess_predictions(  # noqa: SLF001
            outputs_dict, input_data.user_id
        )

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self._cache.clear()
        logger.info("Analysis cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "hit_ratio": self._calculate_hit_ratio(),
            "oldest_entry_age": self._get_oldest_entry_age(),
        }

    @staticmethod
    def _calculate_hit_ratio() -> float:
        """Calculate cache hit ratio (placeholder implementation)."""
        # In a real implementation, you'd track hits/misses
        return 0.0

    def _get_oldest_entry_age(self) -> float:
        """Get age of oldest cache entry in seconds."""
        if not self._cache:
            return 0.0
        oldest_timestamp = min(timestamp for _, timestamp in self._cache.values())
        return time.time() - oldest_timestamp

    async def warm_up(
        self, num_iterations: int = DEFAULT_WARMUP_ITERATIONS
    ) -> dict[str, float]:
        """Warm up the optimized model with dummy data."""
        logger.info("Warming up PAT model with %d iterations...", num_iterations)

        times = []

        for i in range(num_iterations):
            # Create dummy data
            dummy_points = [
                ActigraphyDataPoint(
                    timestamp=datetime.now(UTC) + timedelta(minutes=j),
                    value=30.0 + 20.0 * (i % 2),  # Alternating activity
                )
                for j in range(1440)  # 24 hours of minute-by-minute data
            ]

            dummy_input = ActigraphyInput(
                user_id=f"warmup_user_{i}",
                data_points=dummy_points,
            )

            start_time = time.time()
            await self.optimized_analyze(dummy_input, use_cache=False)
            iteration_time = time.time() - start_time

            times.append(iteration_time)

            logger.info("Warmup iteration %d: %.3fs", i + 1, iteration_time)

        stats = {
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
        }

        logger.info("Warmup completed - Mean: %.3fs", stats["mean_time"])
        return stats


class BatchAnalysisProcessor:
    """Batch processor for handling multiple PAT analysis requests efficiently."""

    def __init__(
        self, optimizer: PATPerformanceOptimizer, max_batch_size: int = MAX_BATCH_SIZE
    ) -> None:
        self.optimizer = optimizer
        self.max_batch_size = max_batch_size
        self.pending_requests: list[
            tuple[ActigraphyInput, asyncio.Future[ActigraphyAnalysis]]
        ] = []
        self.processing = False

    async def analyze_batch(self, input_data: ActigraphyInput) -> ActigraphyAnalysis:
        """Add analysis request to batch and return result when ready."""
        future: asyncio.Future[ActigraphyAnalysis] = asyncio.Future()
        self.pending_requests.append((input_data, future))

        # Trigger batch processing if not already running
        if not self.processing:
            task = asyncio.create_task(self._process_batch())
            # Store reference to prevent garbage collection
            self._background_task = task

        return await future

    async def _process_batch(self) -> None:
        """Process pending requests in batches."""
        self.processing = True

        try:
            while self.pending_requests:
                # Take up to max_batch_size requests
                batch = self.pending_requests[: self.max_batch_size]
                self.pending_requests = self.pending_requests[self.max_batch_size :]

                # Process batch concurrently
                tasks = [
                    self.optimizer.optimized_analyze(input_data, use_cache=True)
                    for input_data, _ in batch
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Set results for futures
                for (_, future), result in zip(batch, results, strict=False):
                    if isinstance(result, Exception):
                        future.set_exception(result)
                    else:
                        analysis, _ = result  # type: ignore[misc]  # Unpack (analysis, was_cached) tuple
                        future.set_result(analysis)

                logger.info("Processed batch of %d requests", len(batch))

        finally:
            self.processing = False


def get_pat_optimizer() -> PATPerformanceOptimizer:
    """Get the global PAT performance optimizer instance."""
    # This would need to be called after the service is loaded
    # In practice, you'd initialize this during app startup
    msg = "Call initialize_pat_optimizer() during app startup"
    raise NotImplementedError(msg)


async def initialize_pat_optimizer() -> PATPerformanceOptimizer:
    """Initialize the PAT performance optimizer during app startup."""
    # Import here to avoid circular dependency
    from clarity.ml.pat_service import get_pat_service  # noqa: PLC0415

    pat_service = await get_pat_service()
    optimizer = PATPerformanceOptimizer(pat_service)

    # Load model if not already loaded
    if not pat_service.is_loaded:
        await pat_service.load_model()

    # Apply default optimizations
    await optimizer.optimize_model(use_torchscript=True, use_pruning=False)

    return optimizer
