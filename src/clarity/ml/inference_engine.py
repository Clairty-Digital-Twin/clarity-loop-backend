"""Asynchronous Inference Engine for PAT Model Service.

This module provides optimized, production-ready inference capabilities for the
Pretrained Actigraphy Transformer (PAT) model, including:

- Asynchronous request processing with batching
- Redis caching for frequent requests
- Performance monitoring and metrics
- Comprehensive error handling and fallback mechanisms
- Resource optimization for concurrent requests

Design Pattern: Producer-Consumer with async queues for efficient batching.
"""

import asyncio
import contextlib
from functools import wraps
import hashlib
import json
import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from clarity.ml.pat_service import ActigraphyAnalysis, ActigraphyInput, PATModelService

logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    """Request for actigraphy inference."""
    request_id: str = Field(..., description="Unique request identifier")
    input_data: ActigraphyInput = Field(..., description="Actigraphy input data")
    priority: int = Field(default=1, description="Request priority (1=highest, 10=lowest)")
    timeout_seconds: float = Field(default=30.0, description="Request timeout")
    cache_enabled: bool = Field(default=True, description="Enable result caching")


class InferenceResponse(BaseModel):
    """Response from inference engine."""
    request_id: str = Field(..., description="Request identifier")
    analysis: ActigraphyAnalysis = Field(..., description="Analysis results")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    cache_hit: bool = Field(default=False, description="Whether result came from cache")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Additional metrics")


class InferenceCache:
    """Simple in-memory cache with TTL support."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self.cache: dict[str, tuple[Any, float]] = {}
        self.ttl = ttl_seconds

    def _cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp > self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]

    async def get(self, key: str) -> dict[str, Any] | None:
        """Get cached value if not expired."""
        self._cleanup_expired()
        if key in self.cache:
            value, _ = self.cache[key]
            return value
        return None

    async def set(self, key: str, value: dict[str, Any]) -> None:
        """Set cached value with timestamp."""
        self.cache[key] = (value, time.time())

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_entries": len(self.cache),
            "ttl_seconds": self.ttl,
            "memory_mb": len(str(self.cache)) / (1024 * 1024)  # Rough estimate
        }


def performance_monitor(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.debug("Function %s took %.2fms", func.__name__, elapsed)
    return wrapper


class AsyncInferenceEngine:
    """High-performance async inference engine for PAT model."""

    def __init__(
        self,
        pat_service: PATModelService,
        batch_size: int = 8,
        batch_timeout_ms: int = 100,
        cache_ttl: int = 300
    ) -> None:
        self.pat_service = pat_service
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout_ms / 1000.0  # Convert to seconds

        # Performance tracking
        self.request_count = 0
        self.cache_hits = 0
        self.error_count = 0

        # Async components
        self.cache = InferenceCache(ttl_seconds=cache_ttl)
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.batch_processor_task: asyncio.Task | None = None
        self.is_running = False

        logger.info(
            "Initialized AsyncInferenceEngine: batch_size=%d, "
            "timeout=%dms, cache_ttl=%ds",
            self.batch_size, batch_timeout_ms, cache_ttl
        )

    async def start(self) -> None:
        """Start the batch processor."""
        if self.is_running:
            return

        self.is_running = True
        self.batch_processor_task = asyncio.create_task(self._batch_processor())
        logger.info("AsyncInferenceEngine started")

    async def stop(self) -> None:
        """Stop the batch processor and cleanup."""
        self.is_running = False

        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.batch_processor_task

        logger.info("AsyncInferenceEngine stopped")

    @staticmethod
    def _generate_cache_key(input_data: ActigraphyInput) -> str:
        """Generate cache key from input data."""
        # Create deterministic hash from input data
        data_dict = {
            "user_id": input_data.user_id,
            "data_points": [
                {
                    "timestamp": dp.timestamp.isoformat(),
                    "value": dp.value,
                }
                for dp in input_data.data_points
            ],
            "sampling_rate": input_data.sampling_rate,
            "duration_hours": input_data.duration_hours
        }
        return hashlib.sha256(
            json.dumps(data_dict, sort_keys=True).encode()
        ).hexdigest()[:16]

    async def _check_cache(self, input_data: ActigraphyInput) -> ActigraphyAnalysis | None:
        """Check cache for existing result."""
        cache_key = self._generate_cache_key(input_data)

        try:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self.cache_hits += 1
                logger.debug("Cache hit for key %s", cache_key)
                return ActigraphyAnalysis(**cached_result)
        except Exception as e:
            logger.warning("Cache check failed: %s", str(e))
        return None

    async def _store_cache(self, input_data: ActigraphyInput, analysis: ActigraphyAnalysis) -> None:
        """Store result in cache."""
        cache_key = self._generate_cache_key(input_data)

        try:
            await self.cache.set(cache_key, analysis.dict())
            logger.debug("Cached result for key %s", cache_key)
        except Exception as e:
            logger.warning("Cache store failed: %s", str(e))

    async def _process_batch(self, requests: list[tuple[InferenceRequest, asyncio.Future]]) -> None:
        """Process a batch of inference requests."""
        if not requests:
            return

        start_time = time.perf_counter()
        batch_size = len(requests)

        logger.info("Processing batch of %d requests", batch_size)

        try:
            # Process each request in the batch
            for request, future in requests:
                if future.cancelled():
                    continue

                try:
                    result = await self._run_single_inference(request)
                    future.set_result(result)
                    self.request_count += 1
                except Exception as e:
                    self.error_count += 1
                    logger.exception("Request %s failed", request.request_id)
                    future.set_exception(e)

        except Exception as e:
            # Batch-level error - fail all requests
            logger.exception("Batch processing failed")
            for _, future in requests:
                if not future.cancelled() and not future.done():
                    future.set_exception(e)

        finally:
            batch_time = (time.perf_counter() - start_time) * 1000
            logger.info("Batch processing completed in %.2fms", batch_time)

    @performance_monitor
    async def _run_single_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference for a single request."""
        # Check cache first if enabled
        if request.cache_enabled:
            cached_result = await self._check_cache(request.input_data)
            if cached_result:
                return InferenceResponse(
                    request_id=request.request_id,
                    analysis=cached_result,
                    processing_time_ms=0.0,
                    cache_hit=True,
                    metrics={"cache_hit": True}
                )

        # Run actual inference
        start_time = time.perf_counter()

        try:
            # Use PAT service for analysis
            analysis = await self.pat_service.analyze_actigraphy(request.input_data)

            processing_time = (time.perf_counter() - start_time) * 1000

            # Store in cache if enabled
            if request.cache_enabled:
                await self._store_cache(request.input_data, analysis)

            return InferenceResponse(
                request_id=request.request_id,
                analysis=analysis,
                processing_time_ms=processing_time,
                cache_hit=False,
                metrics={
                    "cache_hit": False,
                    "confidence": analysis.confidence_score,
                    "processing_time_ms": processing_time
                }
            )

        except Exception as e:
            logger.exception("Inference failed for request %s", request.request_id)
            raise

    async def _batch_processor(self) -> None:
        """Main batch processing loop."""
        while self.is_running:
            try:
                # Collect requests for batching
                requests: list[tuple[InferenceRequest, asyncio.Future]] = []

                # Wait for first request
                try:
                    first_request = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=1.0
                    )
                    requests.append(first_request)
                except TimeoutError:
                    continue

                # Collect additional requests up to batch size
                batch_deadline = time.time() + self.batch_timeout

                while (
                    len(requests) < self.batch_size and
                    time.time() < batch_deadline and
                    self.is_running
                ):
                    try:
                        remaining_time = max(0, batch_deadline - time.time())
                        additional_request = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=remaining_time
                        )
                        requests.append(additional_request)
                    except TimeoutError:
                        break

                # Process the batch
                if requests:
                    await self._process_batch(requests)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Batch processor error")
                await asyncio.sleep(0.1)  # Brief pause on error

    async def predict_async(self, request: InferenceRequest) -> InferenceResponse:
        """Submit inference request for async processing.

        Args:
            request: Inference request with input data and options

        Returns:
            Inference response with analysis and metrics

        Raises:
            asyncio.TimeoutError: If request times out
        """
        if not self.is_running:
            await self.start()

        # Create future for result
        result_future: asyncio.Future = asyncio.Future()

        # Add to queue
        await self.request_queue.put((request, result_future))

        try:
            # Wait for result with timeout
            return await asyncio.wait_for(
                result_future,
                timeout=request.timeout_seconds
            )

        except TimeoutError:
            logger.warning("Request %s timed out", request.request_id)
            result_future.cancel()
            raise

    async def predict(
        self,
        input_data: ActigraphyInput,
        priority: int = 1,
        timeout_seconds: float = 30.0,
        *,
        cache_enabled: bool = True
    ) -> InferenceResponse:
        """Convenient method for single prediction requests.

        Args:
            input_data: Actigraphy input data
            priority: Request priority (1=highest, 10=lowest)
            timeout_seconds: Request timeout
            cache_enabled: Enable result caching

        Returns:
            Inference response with analysis and metrics
        """
        request = InferenceRequest(
            request_id=f"req_{int(time.time() * 1000)}_{id(input_data)}",
            input_data=input_data,
            priority=priority,
            timeout_seconds=timeout_seconds,
            cache_enabled=cache_enabled
        )

        return await self.predict_async(request)

    def get_stats(self) -> dict[str, Any]:
        """Get engine performance statistics."""
        cache_hit_rate = (
            self.cache_hits / max(1, self.request_count) * 100
            if self.request_count > 0 else 0
        )

        return {
            "requests_processed": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "errors": self.error_count,
            "queue_size": self.request_queue.qsize(),
            "is_running": self.is_running
        }


# Global inference engine instance
_inference_engine: AsyncInferenceEngine | None = None


async def get_inference_engine() -> AsyncInferenceEngine:
    """Get or create the global inference engine instance."""
    global _inference_engine

    if _inference_engine is None:
        from clarity.ml.pat_service import get_pat_service
        pat_service = await get_pat_service()
        _inference_engine = AsyncInferenceEngine(pat_service)
        await _inference_engine.start()

    return _inference_engine


async def shutdown_inference_engine() -> None:
    """Shutdown the global inference engine."""
    global _inference_engine

    if _inference_engine:
        await _inference_engine.stop()
        _inference_engine = None
