"""High-performance async inference engine for PAT model analysis.

This module provides a production-ready inference engine with:
- Async batch processing for optimal throughput
- Intelligent caching with TTL support
- Performance monitoring and metrics
- Graceful error handling and recovery
- Request queuing and timeout management

The engine is designed to handle high-volume actigraphy analysis requests
while maintaining low latency and high reliability.
"""

import asyncio
from collections.abc import Callable
import contextlib
from functools import wraps
import hashlib
import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from clarity.ml.pat_service import ActigraphyAnalysis, ActigraphyInput, PATModelService

logger = logging.getLogger(__name__)

# Global inference engine instance
_inference_engine: "AsyncInferenceEngine | None" = None


class InferenceRequest(BaseModel):
    """Request for PAT model inference."""

    request_id: str = Field(description="Unique request identifier")
    input_data: ActigraphyInput = Field(description="Actigraphy input data")
    timeout_seconds: float = Field(default=30.0, description="Request timeout")
    cache_enabled: bool = Field(default=True, description="Enable result caching")


class InferenceResponse(BaseModel):
    """Response from PAT model inference."""

    request_id: str = Field(description="Request identifier")
    analysis: ActigraphyAnalysis = Field(description="Analysis results")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    cached: bool = Field(default=False, description="Whether result was cached")
    timestamp: float = Field(description="Response timestamp")


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

    async def get(self, key: str) -> Any | None:
        """Get value from cache if not expired."""
        self._cleanup_expired()
        if key in self.cache:
            value, _ = self.cache[key]
            return value
        return None

    async def set(self, key: str, value: Any) -> None:
        """Set value in cache with current timestamp."""
        self.cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()


def performance_monitor(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to monitor function performance."""
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            duration = (time.perf_counter() - start_time) * 1000
            logger.debug("Function %s completed in %.2fms", func.__name__, duration)
            return result
        except Exception as e:
            duration = (time.perf_counter() - start_time) * 1000
            logger.warning("Function %s failed after %.2fms: %s", func.__name__, duration, str(e))
            raise
    return wrapper


class AsyncInferenceEngine:
    """High-performance async inference engine for PAT model."""

    def __init__(
        self,
        pat_service: PATModelService,
        cache_ttl: int = 300,
        batch_size: int = 4,
        batch_timeout_ms: int = 100
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
        self.request_queue: asyncio.Queue[tuple[InferenceRequest, asyncio.Future[InferenceResponse]]] = asyncio.Queue()
        self.batch_processor_task: asyncio.Task[None] | None = None
        self.is_running = False

        logger.info(
            "Initialized AsyncInferenceEngine: batch_size=%d, "
            "cache_ttl=%ds, batch_timeout=%.1fms",
            batch_size, cache_ttl, batch_timeout_ms
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

    def _generate_cache_key(self, input_data: ActigraphyInput) -> str:
        """Generate cache key for input data."""
        # Create a hash of the input data for caching
        data_str = f"{input_data.user_id}_{len(input_data.data_points)}_{input_data.sampling_rate}"

        # Add first and last data point values for uniqueness
        if input_data.data_points:
            first_point = input_data.data_points[0]
            last_point = input_data.data_points[-1]
            data_str += f"_{first_point.timestamp}_{first_point.value}_{last_point.timestamp}_{last_point.value}"

        return hashlib.md5(data_str.encode()).hexdigest()

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

    async def _process_batch(self, requests: list[tuple[InferenceRequest, asyncio.Future[InferenceResponse]]]) -> None:
        """Process a batch of inference requests."""
        if not requests:
            return

        start_time = time.perf_counter()
        batch_size = len(requests)
        logger.debug("Processing batch of %d requests", batch_size)

        # Process each request in the batch
        for request, future in requests:
            if future.cancelled():
                continue

            try:
                result = await self._run_single_inference(request)
                if not future.cancelled():
                    future.set_result(result)
            except Exception as e:
                if not future.cancelled():
                    future.set_exception(e)

        processing_time = (time.perf_counter() - start_time) * 1000
        logger.debug("Batch processing completed in %.2fms", processing_time)

    @performance_monitor
    async def _run_single_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference for a single request."""
        start_time = time.perf_counter()
        self.request_count += 1

        try:
            # Check cache first if enabled
            cached_result = None
            if request.cache_enabled:
                cached_result = await self._check_cache(request.input_data)

            if cached_result:
                processing_time = (time.perf_counter() - start_time) * 1000
                return InferenceResponse(
                    request_id=request.request_id,
                    analysis=cached_result,
                    processing_time_ms=processing_time,
                    cached=True,
                    timestamp=time.time()
                )

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
                cached=False,
                timestamp=time.time()
            )

        except Exception:
            logger.exception("Inference failed for request %s", request.request_id)
            raise

    async def _batch_processor(self) -> None:
        """Main batch processing loop."""
        logger.info("Batch processor started")

        while self.is_running:
            try:
                # Collect requests for batching
                requests: list[tuple[InferenceRequest, asyncio.Future[InferenceResponse]]] = []

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
                    time.time() < batch_deadline
                ):
                    try:
                        additional_request = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=max(0.001, batch_deadline - time.time())
                        )
                        requests.append(additional_request)
                    except TimeoutError:
                        break

                # Process the batch
                if requests:
                    await self._process_batch(requests)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Batch processor error")
                await asyncio.sleep(0.1)  # Brief pause on error

        logger.info("Batch processor stopped")

    async def predict_async(self, request: InferenceRequest) -> InferenceResponse:
        """Submit async prediction request."""
        # Ensure engine is running
        if not self.is_running:
            await self.start()

        # Create future for result
        result_future: asyncio.Future[InferenceResponse] = asyncio.Future()

        # Add to queue
        await self.request_queue.put((request, result_future))

        try:
            # Wait for result with timeout
            return await asyncio.wait_for(result_future, timeout=request.timeout_seconds)
        except TimeoutError as e:
            result_future.cancel()
            msg = f"Request {request.request_id} timed out after {request.timeout_seconds}s"
            raise TimeoutError(msg) from e

    async def predict(
        self,
        input_data: ActigraphyInput,
        request_id: str,
        timeout_seconds: float = 30.0,
        cache_enabled: bool = True
    ) -> InferenceResponse:
        """Convenience method for single prediction."""
        request = InferenceRequest(
            request_id=request_id,
            input_data=input_data,
            timeout_seconds=timeout_seconds,
            cache_enabled=cache_enabled
        )

        return await self.predict_async(request)

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        cache_hit_rate = (
            self.cache_hits / self.request_count * 100
            if self.request_count > 0 else 0
        )

        return {
            "requests_processed": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate_percent": cache_hit_rate,
            "error_count": self.error_count,
            "is_running": self.is_running,
            "queue_size": self.request_queue.qsize() if hasattr(self.request_queue, 'qsize') else 0
        }


# Global inference engine management
async def get_inference_engine() -> AsyncInferenceEngine:
    """Get or create the global inference engine instance."""
    global _inference_engine  # noqa: PLW0603

    if _inference_engine is None:
        from clarity.ml.pat_service import get_pat_service  # noqa: PLC0415
        pat_service = await get_pat_service()
        _inference_engine = AsyncInferenceEngine(pat_service)
        await _inference_engine.start()

    return _inference_engine


async def shutdown_inference_engine() -> None:
    """Shutdown the global inference engine."""
    global _inference_engine  # noqa: PLW0603

    if _inference_engine:
        await _inference_engine.stop()
        _inference_engine = None
        logger.info("Global inference engine shutdown complete")


# Dependency injection helper
async def get_pat_inference_engine() -> AsyncInferenceEngine:
    """FastAPI dependency to get the PAT inference engine."""
    return await get_inference_engine()
