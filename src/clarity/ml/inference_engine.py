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
import hashlib
import json
import logging
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from functools import wraps

import numpy as np
import torch
from pydantic import BaseModel, Field

from clarity.ml.pat_service import ActigraphyInput, ActigraphyAnalysis, PATModelService
from clarity.ml.preprocessing import ActigraphyDataPoint

logger = logging.getLogger(__name__)

# Performance constants
DEFAULT_BATCH_SIZE = 8
MAX_BATCH_SIZE = 32
BATCH_TIMEOUT_MS = 100  # Maximum time to wait for batch completion
DEFAULT_CACHE_TTL = 3600  # 1 hour cache TTL
MAX_CONCURRENT_REQUESTS = 100


@dataclass
class InferenceMetrics:
    """Performance metrics for inference operations."""
    request_id: str
    batch_size: int
    inference_time_ms: float
    preprocessing_time_ms: float
    postprocessing_time_ms: float
    cache_hit: bool
    model_confidence: float
    total_time_ms: float


class InferenceRequest(BaseModel):
    """Request wrapper for async inference processing."""
    request_id: str
    input_data: ActigraphyInput
    priority: int = Field(default=1, description="Higher numbers = higher priority")
    timeout_seconds: float = Field(default=30.0, description="Request timeout")
    cache_enabled: bool = Field(default=True, description="Enable result caching")
    
    
class InferenceResponse(BaseModel):
    """Response wrapper with metadata and performance metrics."""
    request_id: str
    analysis: ActigraphyAnalysis
    metrics: Dict[str, Any]
    cached: bool = Field(default=False, description="Result was cached")
    processing_time_ms: float
    

class BatchProcessingError(Exception):
    """Exception raised during batch processing."""
    pass


class InferenceCache:
    """Redis-like cache interface for inference results.
    
    Note: This is a simplified in-memory implementation.
    In production, replace with actual Redis client.
    """
    
    def __init__(self, ttl_seconds: int = DEFAULT_CACHE_TTL):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.ttl_seconds = ttl_seconds
        
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired."""
        return time.time() - timestamp > self.ttl_seconds
        
    def _cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if self._is_expired(timestamp)
        ]
        for key in expired_keys:
            del self.cache[key]
            
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        self._cleanup_expired()
        if key in self.cache:
            value, timestamp = self.cache[key]
            if not self._is_expired(timestamp):
                return value
            else:
                del self.cache[key]
        return None
        
    async def set(self, key: str, value: Any) -> None:
        """Set cached value with timestamp."""
        self.cache[key] = (value, time.time())
        
    async def delete(self, key: str) -> None:
        """Delete cached value."""
        self.cache.pop(key, None)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        self._cleanup_expired()
        return {
            "total_entries": len(self.cache),
            "ttl_seconds": self.ttl_seconds,
            "memory_mb": len(str(self.cache)) / (1024 * 1024)  # Rough estimate
        }


def performance_monitor(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.debug(f"{func.__name__} took {elapsed:.2f}ms")
    return wrapper


class AsyncInferenceEngine:
    """Production-ready asynchronous inference engine for PAT model.
    
    Features:
    - Efficient request batching with configurable timeouts
    - Redis-compatible caching with TTL support
    - Performance monitoring and metrics collection
    - Graceful error handling and circuit breaker patterns
    - Resource management and concurrent request limiting
    """
    
    def __init__(
        self,
        pat_service: PATModelService,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_timeout_ms: int = BATCH_TIMEOUT_MS,
        cache_ttl: int = DEFAULT_CACHE_TTL,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
    ):
        self.pat_service = pat_service
        self.batch_size = min(batch_size, MAX_BATCH_SIZE)
        self.batch_timeout = batch_timeout_ms / 1000.0  # Convert to seconds
        self.max_concurrent = max_concurrent
        
        # Initialize cache
        self.cache = InferenceCache(ttl_seconds=cache_ttl)
        
        # Request queue and batch processing
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.batch_processor_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Performance tracking
        self.total_requests = 0
        self.cache_hits = 0
        self.batch_count = 0
        self.error_count = 0
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.info(
            f"Initialized AsyncInferenceEngine: batch_size={self.batch_size}, "
            f"timeout={batch_timeout_ms}ms, cache_ttl={cache_ttl}s"
        )
        
    async def start(self) -> None:
        """Start the batch processing engine."""
        if self.is_running:
            return
            
        self.is_running = True
        self.batch_processor_task = asyncio.create_task(self._batch_processor())
        logger.info("AsyncInferenceEngine started")
        
    async def stop(self) -> None:
        """Stop the batch processing engine."""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
                
        logger.info("AsyncInferenceEngine stopped")
        
    def _generate_cache_key(self, input_data: ActigraphyInput) -> str:
        """Generate cache key from input data."""
        # Create deterministic hash from input data
        data_dict = {
            "user_id": input_data.user_id,
            "data_points": [
                {
                    "timestamp": dp.timestamp.isoformat(),
                    "activity_counts": dp.activity_counts,
                    "steps": dp.steps,
                    "heart_rate": dp.heart_rate
                }
                for dp in input_data.data_points
            ],
            "sampling_rate": input_data.sampling_rate,
            "duration_hours": input_data.duration_hours
        }
        
        json_str = json.dumps(data_dict, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
        
    @performance_monitor
    async def _check_cache(self, cache_key: str) -> Optional[ActigraphyAnalysis]:
        """Check cache for existing result."""
        try:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self.cache_hits += 1
                logger.debug(f"Cache hit for key {cache_key}")
                return ActigraphyAnalysis(**cached_result)
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
        return None
        
    @performance_monitor  
    async def _store_cache(self, cache_key: str, analysis: ActigraphyAnalysis) -> None:
        """Store result in cache."""
        try:
            await self.cache.set(cache_key, analysis.dict())
            logger.debug(f"Cached result for key {cache_key}")
        except Exception as e:
            logger.warning(f"Cache store failed: {e}")
            
    async def _process_batch(self, requests: List[Tuple[InferenceRequest, asyncio.Future]]) -> None:
        """Process a batch of inference requests."""
        if not requests:
            return
            
        start_time = time.perf_counter()
        self.batch_count += 1
        batch_size = len(requests)
        
        logger.info(f"Processing batch of {batch_size} requests")
        
        try:
            # Process each request in the batch
            for request, future in requests:
                if future.cancelled():
                    continue
                    
                try:
                    # Individual request processing
                    result = await self._process_single_request(request)
                    future.set_result(result)
                    
                except Exception as e:
                    self.error_count += 1
                    logger.error(f"Request {request.request_id} failed: {e}")
                    future.set_exception(e)
                    
        except Exception as e:
            # Batch-level error - fail all requests
            logger.error(f"Batch processing failed: {e}")
            for _, future in requests:
                if not future.cancelled() and not future.done():
                    future.set_exception(BatchProcessingError(f"Batch processing failed: {e}"))
                    
        finally:
            batch_time = (time.perf_counter() - start_time) * 1000
            logger.info(f"Batch processing completed in {batch_time:.2f}ms")
            
    @performance_monitor
    async def _process_single_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process a single inference request with caching."""
        start_time = time.perf_counter()
        cache_key = self._generate_cache_key(request.input_data)
        
        # Check cache first
        if request.cache_enabled:
            cached_analysis = await self._check_cache(cache_key)
            if cached_analysis:
                processing_time = (time.perf_counter() - start_time) * 1000
                return InferenceResponse(
                    request_id=request.request_id,
                    analysis=cached_analysis,
                    metrics={"cache_hit": True},
                    cached=True,
                    processing_time_ms=processing_time
                )
                
        # Run actual inference
        preprocessing_start = time.perf_counter()
        
        try:
            # Perform analysis using PAT service
            analysis = await self.pat_service.analyze_actigraphy(request.input_data)
            
            # Cache the result
            if request.cache_enabled:
                await self._store_cache(cache_key, analysis)
                
            processing_time = (time.perf_counter() - start_time) * 1000
            
            metrics = {
                "cache_hit": False,
                "confidence": analysis.confidence_score,
                "processing_time_ms": processing_time
            }
            
            return InferenceResponse(
                request_id=request.request_id,
                analysis=analysis,
                metrics=metrics,
                cached=False,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Inference failed for request {request.request_id}: {e}")
            raise
            
    async def _batch_processor(self) -> None:
        """Main batch processing loop."""
        logger.info("Batch processor started")
        
        while self.is_running:
            try:
                # Collect requests for batching
                requests: List[Tuple[InferenceRequest, asyncio.Future]] = []
                
                # Wait for first request
                try:
                    first_item = await asyncio.wait_for(
                        self.request_queue.get(), 
                        timeout=1.0
                    )
                    requests.append(first_item)
                except asyncio.TimeoutError:
                    continue
                    
                # Collect additional requests up to batch size or timeout
                batch_start = time.perf_counter()
                while (
                    len(requests) < self.batch_size 
                    and (time.perf_counter() - batch_start) < self.batch_timeout
                ):
                    try:
                        remaining_timeout = self.batch_timeout - (time.perf_counter() - batch_start)
                        if remaining_timeout <= 0:
                            break
                            
                        item = await asyncio.wait_for(
                            self.request_queue.get(), 
                            timeout=remaining_timeout
                        )
                        requests.append(item)
                    except asyncio.TimeoutError:
                        break
                        
                # Process the batch
                if requests:
                    await self._process_batch(requests)
                    
            except asyncio.CancelledError:
                logger.info("Batch processor cancelled")
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error
                
        logger.info("Batch processor stopped")
        
    async def predict_async(self, request: InferenceRequest) -> InferenceResponse:
        """Submit inference request for async processing.
        
        Args:
            request: Inference request with input data and options
            
        Returns:
            Inference response with analysis and metrics
            
        Raises:
            asyncio.TimeoutError: If request times out
            BatchProcessingError: If batch processing fails
        """
        # Concurrency control
        async with self.semaphore:
            self.total_requests += 1
            
            # Ensure engine is running
            if not self.is_running:
                await self.start()
                
            # Create future for result
            result_future: asyncio.Future = asyncio.Future()
            
            # Add to queue
            await self.request_queue.put((request, result_future))
            
            try:
                # Wait for result with timeout
                result = await asyncio.wait_for(
                    result_future, 
                    timeout=request.timeout_seconds
                )
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Request {request.request_id} timed out")
                result_future.cancel()
                raise
                
    async def predict(
        self,
        input_data: ActigraphyInput,
        request_id: Optional[str] = None,
        priority: int = 1,
        timeout_seconds: float = 30.0,
        cache_enabled: bool = True
    ) -> InferenceResponse:
        """Convenient method for single prediction requests.
        
        Args:
            input_data: Actigraphy input data
            request_id: Optional request identifier (auto-generated if None)
            priority: Request priority (higher = more important)
            timeout_seconds: Request timeout
            cache_enabled: Enable result caching
            
        Returns:
            Inference response with analysis and metrics
        """
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000)}_{self.total_requests}"
            
        request = InferenceRequest(
            request_id=request_id,
            input_data=input_data,
            priority=priority,
            timeout_seconds=timeout_seconds,
            cache_enabled=cache_enabled
        )
        
        return await self.predict_async(request)
        
    async def health_check(self) -> Dict[str, Any]:
        """Get health status and performance metrics."""
        cache_stats = self.cache.get_stats()
        
        return {
            "service": "AsyncInferenceEngine",
            "status": "healthy" if self.is_running else "stopped",
            "performance": {
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": self.cache_hits / max(self.total_requests, 1),
                "batch_count": self.batch_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.total_requests, 1),
            },
            "configuration": {
                "batch_size": self.batch_size,
                "batch_timeout_ms": self.batch_timeout * 1000,
                "max_concurrent": self.max_concurrent,
                "queue_size": self.request_queue.qsize(),
            },
            "cache": cache_stats,
            "pat_service": await self.pat_service.health_check()
        }


# Global inference engine instance
_inference_engine: Optional[AsyncInferenceEngine] = None


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