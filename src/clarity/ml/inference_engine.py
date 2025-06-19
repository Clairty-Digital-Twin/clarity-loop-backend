"""High-performance async inference engine for PAT model analysis.

This module provides a production-ready inference engine with:
- Async batch processing for optimal throughput
- Intelligent caching with TTL support
- Performance monitoring and metrics
- Graceful error handling and recovery
- Request queuing and timeout management

The engine is designed to handle high-volume actigraphy analysis requests
while maintaining low latency and high reliability.

This implementation follows clean code principles with proper separation of concerns,
dependency injection, and comprehensive error handling.
"""

# removed - breaks FastAPI

import asyncio
from collections.abc import Callable
from functools import wraps
import logging
import time
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, Field

from clarity.core.constants import (
    BATCH_PROCESSOR_ERROR_SLEEP_SECONDS,
    CACHE_TTL_DEFAULT_SECONDS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_TIMEOUT_MS,
    DEFAULT_INFERENCE_TIMEOUT_SECONDS,
)
from clarity.core.exceptions import InferenceError, InferenceTimeoutError
from clarity.core.security import create_secure_cache_key
from clarity.core.types import CacheStorage, LoggerProtocol
from clarity.ml.pat_service import (
    ActigraphyAnalysis,
    ActigraphyInput,
    PATModelService,
    get_pat_service,
)
from clarity.utils.decorators import resilient_prediction

if TYPE_CHECKING:
    pass  # Only for type stubs now

logger: LoggerProtocol = logging.getLogger(__name__)

# Global inference engine instance
_inference_engine: "AsyncInferenceEngine | None" = None


class InferenceRequest(BaseModel):
    """Request for PAT model inference."""

    request_id: str = Field(description="Unique request identifier")
    input_data: ActigraphyInput = Field(description="Actigraphy input data")
    timeout_seconds: float = Field(
        default=DEFAULT_INFERENCE_TIMEOUT_SECONDS, description="Request timeout"
    )
    cache_enabled: bool = Field(default=True, description="Enable result caching")


class InferenceResponse(BaseModel):
    """Response from PAT model inference."""

    request_id: str = Field(description="Request identifier")
    analysis: ActigraphyAnalysis = Field(description="Analysis results")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    cached: bool = Field(default=False, description="Whether result was cached")
    timestamp: float = Field(description="Response timestamp")


class InferenceCache:
    """Simple in-memory cache with TTL support.

    This class provides efficient caching with automatic cleanup of expired entries,
    following the Single Responsibility Principle.
    """

    def __init__(self, ttl_seconds: int = CACHE_TTL_DEFAULT_SECONDS) -> None:
        """Initialize cache with specified TTL.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.cache: CacheStorage = {}
        self.ttl = ttl_seconds

    def _cleanup_expired(self) -> None:
        """Remove expired entries from cache.

        This method implements the Strategy pattern for cache cleanup,
        allowing for different cleanup strategies in the future.
        """
        current_time = time.time()
        expired_keys = [
            key
            for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp > self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]

    async def get(self, key: str) -> object | None:
        """Get value from cache if not expired.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value if found and not expired, None otherwise
        """
        self._cleanup_expired()
        if key in self.cache:
            value, _ = self.cache[key]
            return value  # type: ignore[no-any-return]  # Cache preserves original function's return type
        return None

    async def set(self, key: str, value: object) -> None:
        """Set value in cache with current timestamp.

        Args:
            key: Cache key to set
            value: Value to cache
        """
        self.cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()


def performance_monitor(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to monitor function performance.

    This decorator implements the Decorator pattern to add performance
    monitoring capabilities to any async function without modifying its code.

    Args:
        func: Function to monitor

    Returns:
        Wrapped function with performance monitoring
    """

    @wraps(func)
    async def wrapper(*args: object, **kwargs: object) -> object:
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            duration = (time.perf_counter() - start_time) * 1000
            logger.debug("Function %s completed in %.2fms", func.__name__, duration)
        except Exception as e:
            duration = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "Function %s failed after %.2fms: %s", func.__name__, duration, str(e)
            )
            raise
        else:
            return result

    return wrapper


class AsyncInferenceEngine:
    """High-performance async inference engine for PAT model.

    This class implements the Facade pattern, providing a simplified interface
    to the complex subsystem of PAT model inference, caching, and batch processing.

    It also follows the Builder pattern for configuration and the Observer pattern
    for performance monitoring.
    """

    def __init__(
        self,
        pat_service: PATModelService,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_timeout_ms: int = DEFAULT_BATCH_TIMEOUT_MS,
        cache_ttl: int = CACHE_TTL_DEFAULT_SECONDS,
    ) -> None:
        """Initialize the inference engine.

        Args:
            pat_service: PAT model service instance
            batch_size: Maximum batch size for processing
            batch_timeout_ms: Batch timeout in milliseconds
            cache_ttl: Cache time-to-live in seconds
        """
        self.pat_service = pat_service
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout_ms / 1000.0  # Convert to seconds

        # Statistics
        self.request_count = 0
        self.cache_hits = 0
        self.error_count = 0

        # Async components
        self.cache = InferenceCache(ttl_seconds=cache_ttl)
        self.request_queue: asyncio.Queue[
            tuple[InferenceRequest, asyncio.Future[InferenceResponse]]
        ] = asyncio.Queue()
        self.batch_processor_task: asyncio.Task[None] | None = None
        self.is_running = False
        self._shutdown_event = asyncio.Event()

        logger.info(
            "Initialized AsyncInferenceEngine: batch_size=%d, "
            "cache_ttl=%ds, batch_timeout=%.1fms",
            batch_size,
            cache_ttl,
            batch_timeout_ms,
        )

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit with proper cleanup."""
        await self.stop()

    async def start(self) -> None:
        """Start the batch processor.

        This method implements the Command pattern to encapsulate
        the startup operation.
        """
        if self.is_running:
            return

        self.is_running = True
        self._shutdown_event.clear()
        self.batch_processor_task = asyncio.create_task(self._batch_processor())
        logger.info("AsyncInferenceEngine started")

    async def stop(self) -> None:
        """Stop the batch processor and cleanup.

        This method implements graceful shutdown with proper resource cleanup.
        """
        self.is_running = False
        self._shutdown_event.set()

        if self.batch_processor_task and not self.batch_processor_task.done():
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass  # Expected when cancelling
            except (RuntimeError, OSError) as e:
                logger.warning("Error during batch processor shutdown: %s", e)

        # Clear any remaining items in the queue
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info("AsyncInferenceEngine stopped")

    @staticmethod
    def _generate_cache_key(input_data: ActigraphyInput) -> str:
        """Generate secure cache key for input data.

        This static method follows the Factory Method pattern to create
        cache keys with consistent format and security.

        Args:
            input_data: Actigraphy input data

        Returns:
            Secure cache key string
        """
        # Create components for the cache key
        data_signature = f"{input_data.user_id}_{len(input_data.data_points)}_{input_data.sampling_rate}"

        # Add first and last data point values for uniqueness
        if input_data.data_points:
            first_point = input_data.data_points[0]
            last_point = input_data.data_points[-1]
            data_signature += f"_{first_point.timestamp}_{first_point.value}_{last_point.timestamp}_{last_point.value}"

        return create_secure_cache_key(data_signature)

    async def _check_cache(
        self, input_data: ActigraphyInput
    ) -> ActigraphyAnalysis | None:
        """Check cache for existing result.

        Args:
            input_data: Input data to check cache for

        Returns:
            Cached analysis result if found, None otherwise
        """
        cache_key = self._generate_cache_key(input_data)

        try:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self.cache_hits += 1
                logger.debug("Cache hit for key %s", cache_key)
                return ActigraphyAnalysis(**cached_result)  # type: ignore[arg-type]
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Cache check failed: %s", str(e))
        return None

    async def _store_cache(
        self, input_data: ActigraphyInput, analysis: ActigraphyAnalysis
    ) -> None:
        """Store result in cache.

        Args:
            input_data: Input data used as cache key basis
            analysis: Analysis result to cache
        """
        cache_key = self._generate_cache_key(input_data)

        try:
            await self.cache.set(cache_key, analysis.dict())
            logger.debug("Cached result for key %s", cache_key)
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Cache store failed: %s", str(e))

    async def _process_batch(
        self, requests: list[tuple[InferenceRequest, asyncio.Future[InferenceResponse]]]
    ) -> None:
        """Process a batch of inference requests.

        This method handles the actual ML inference for a batch of requests,
        implementing caching and error handling.

        Args:
            requests: List of request/future pairs to process
        """
        if not requests:
            return

        start_time = time.perf_counter()
        logger.debug("Processing batch of %d requests", len(requests))

        # Process each request in the batch
        for request, future in requests:
            if future.cancelled():
                continue

            try:
                result = await self._run_single_inference(request)
                if not future.cancelled():
                    future.set_result(result)
            except (ValueError, RuntimeError, OSError) as e:
                self.error_count += 1
                if not future.cancelled():
                    future.set_exception(e)

        processing_time = (time.perf_counter() - start_time) * 1000
        logger.debug("Batch processed in %.2fms", processing_time)

    @performance_monitor
    async def _run_single_inference(
        self, request: InferenceRequest
    ) -> InferenceResponse:
        """Run inference for a single request.

        This method implements the Chain of Responsibility pattern for
        processing steps: cache check -> model inference -> cache store.

        Args:
            request: Inference request to process

        Returns:
            Inference response with analysis results

        Raises:
            InferenceError: If inference fails
        """
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
                    timestamp=time.time(),
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
                timestamp=time.time(),
            )

        except Exception as e:
            logger.exception("Inference failed for request %s", request.request_id)
            error_message = f"Inference failed: {e!s}"
            raise InferenceError(error_message, request_id=request.request_id) from e

    async def _batch_processor(self) -> None:
        """Process inference requests in batches."""
        while self.is_running:
            try:
                # Check for shutdown signal with timeout
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=self.batch_timeout
                    )
                    # Shutdown signal received
                    break
                except TimeoutError:
                    # Continue processing
                    pass

                # Collect requests for batching
                requests: list[
                    tuple[InferenceRequest, asyncio.Future[InferenceResponse]]
                ] = []

                # Try to get first request with timeout
                try:
                    first_request = await asyncio.wait_for(
                        self.request_queue.get(), timeout=self.batch_timeout
                    )
                    requests.append(first_request)
                except TimeoutError:
                    continue  # No requests, continue loop

                # Collect additional requests up to batch size
                for _ in range(self.batch_size - 1):
                    try:
                        additional_request = await asyncio.wait_for(
                            self.request_queue.get(), timeout=self.batch_timeout
                        )
                        requests.append(additional_request)
                    except TimeoutError:
                        break  # No more requests, process current batch

                # Process the batch
                await self._process_batch(requests)

            except asyncio.CancelledError:
                # Graceful shutdown
                logger.info("Batch processor cancelled, shutting down gracefully")
                break
            except Exception:
                logger.exception("Batch processor error")
                self.error_count += 1
                # Don't sleep on shutdown
                if self.is_running:
                    try:
                        await asyncio.wait_for(
                            asyncio.sleep(BATCH_PROCESSOR_ERROR_SLEEP_SECONDS),
                            timeout=1.0,
                        )
                    except TimeoutError:
                        break  # Shutdown timeout reached

    @resilient_prediction(model_name="InferenceEngine")
    async def predict_async(self, request: InferenceRequest) -> InferenceResponse:
        """Process inference request asynchronously with batching.

        Args:
            request: Inference request to process

        Returns:
            Inference response with analysis results

        Raises:
            InferenceTimeoutError: If request times out
            RuntimeError: If engine is not running
        """
        if not self.is_running:
            await self.start()

        # Create future for result
        result_future: asyncio.Future[InferenceResponse] = asyncio.Future()

        # Add to queue
        await self.request_queue.put((request, result_future))

        try:
            # Wait for result with timeout
            return await asyncio.wait_for(
                result_future, timeout=request.timeout_seconds
            )
        except TimeoutError as e:
            result_future.cancel()
            raise InferenceTimeoutError(
                request.request_id, request.timeout_seconds
            ) from e

    async def predict(
        self,
        input_data: ActigraphyInput,
        request_id: str,
        *,
        timeout_seconds: float = DEFAULT_INFERENCE_TIMEOUT_SECONDS,
        cache_enabled: bool = True,
    ) -> InferenceResponse:
        """Convenience method for single prediction.

        Args:
            input_data: Actigraphy input data
            request_id: Unique request identifier
            timeout_seconds: Request timeout in seconds
            cache_enabled: Whether to enable caching

        Returns:
            Inference response with results
        """
        request = InferenceRequest(
            request_id=request_id,
            input_data=input_data,
            timeout_seconds=timeout_seconds,
            cache_enabled=cache_enabled,
        )

        return await self.predict_async(request)

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics.

        Returns:
            Dictionary containing performance metrics
        """
        cache_hit_rate = (
            self.cache_hits / self.request_count * 100 if self.request_count > 0 else 0
        )

        return {
            "requests_processed": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate_percent": cache_hit_rate,
            "error_count": self.error_count,
            "is_running": self.is_running,
            "queue_size": (
                self.request_queue.qsize()
                if hasattr(self.request_queue, "qsize")
                else 0
            ),
        }


# Global inference engine management
async def get_inference_engine() -> AsyncInferenceEngine:
    """Get or create the global inference engine instance.

    This function implements the Singleton pattern to ensure only one
    inference engine instance exists globally.

    Returns:
        Global inference engine instance
    """
    global _inference_engine  # noqa: PLW0603 - Singleton pattern for ML inference engine

    if _inference_engine is None:
        pat_service = await get_pat_service()
        _inference_engine = AsyncInferenceEngine(pat_service)
        await _inference_engine.start()

    return _inference_engine


async def shutdown_inference_engine() -> None:
    """Shutdown the global inference engine.

    This function provides clean shutdown capabilities for the global
    inference engine instance.
    """
    global _inference_engine  # noqa: PLW0603 - Singleton pattern for ML inference engine

    if _inference_engine:
        await _inference_engine.stop()
        _inference_engine = None
        logger.info("Global inference engine shutdown complete")


# Dependency injection helper
async def get_pat_inference_engine() -> AsyncInferenceEngine:
    """FastAPI dependency to get the PAT inference engine.

    This function serves as a dependency provider for FastAPI endpoints,
    implementing the Dependency Injection pattern.

    Returns:
        PAT inference engine instance
    """
    return await get_inference_engine()
