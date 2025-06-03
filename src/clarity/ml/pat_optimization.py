"""Performance optimization utilities for PAT service.

This module provides performance enhancements for the PAT model service including:
- Model optimization (TorchScript compilation)
- Inference batching and caching
- Memory management
- Scalability improvements
"""

import asyncio
from functools import lru_cache
import hashlib
import logging
from pathlib import Path
import time
from typing import Any, Optional

import torch
import torch.jit
from torch.nn.utils import prune

from clarity.ml.pat_service import PATModelService, ActigraphyInput, ActigraphyAnalysis

logger = logging.getLogger(__name__)


class PATPerformanceOptimizer:
    """Performance optimizer for PAT model service."""

    def __init__(self, pat_service: PATModelService):
        self.pat_service = pat_service
        self.compiled_model: Optional[torch.jit.ScriptModule] = None
        self.optimization_enabled = False
        self._cache: dict[str, tuple[ActigraphyAnalysis, float]] = {}
        self._cache_ttl = 3600  # 1 hour cache TTL

    async def optimize_model(
        self,
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
        try:
            if not self.pat_service.is_loaded or not self.pat_service.model:
                logger.error("PAT model not loaded, cannot optimize")
                return False

            logger.info("Starting PAT model optimization...")
            
            model = self.pat_service.model
            
            # Apply pruning if requested
            if use_pruning:
                logger.info(f"Applying structured pruning (amount: {pruning_amount})")
                self._apply_model_pruning(model, pruning_amount)
            
            # Compile with TorchScript if requested
            if use_torchscript:
                logger.info("Compiling model with TorchScript...")
                self.compiled_model = await self._compile_torchscript(model)
                
                if self.compiled_model:
                    logger.info("TorchScript compilation successful")
                    # Optionally save compiled model
                    await self._save_compiled_model()
                else:
                    logger.warning("TorchScript compilation failed, using eager mode")
            
            self.optimization_enabled = True
            logger.info("PAT model optimization completed")
            return True
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return False

    def _apply_model_pruning(self, model: torch.nn.Module, amount: float) -> None:
        """Apply structured pruning to reduce model size."""
        # Prune attention layers
        for layer in model.encoder.transformer_layers:
            # Prune query/key/value projections
            for head_idx in range(layer.attention.num_heads):
                prune.l1_unstructured(
                    layer.attention.query_projections[head_idx], 
                    name='weight', 
                    amount=amount
                )
                prune.l1_unstructured(
                    layer.attention.key_projections[head_idx], 
                    name='weight', 
                    amount=amount
                )
                prune.l1_unstructured(
                    layer.attention.value_projections[head_idx], 
                    name='weight', 
                    amount=amount
                )
            
            # Prune feed-forward layers
            prune.l1_unstructured(layer.ff1, name='weight', amount=amount)
            prune.l1_unstructured(layer.ff2, name='weight', amount=amount)

    async def _compile_torchscript(
        self, model: torch.nn.Module
    ) -> Optional[torch.jit.ScriptModule]:
        """Compile model with TorchScript for optimization."""
        try:
            # Create sample input for tracing
            sample_input = torch.randn(1, 10080).to(self.pat_service.device)
            
            # Use torch.jit.trace for better performance
            with torch.no_grad():
                traced_model = torch.jit.trace(model, sample_input, strict=False)
            
            # Optimize the traced model
            traced_model = torch.jit.optimize_for_inference(traced_model)
            
            return traced_model
            
        except Exception as e:
            logger.error(f"TorchScript compilation failed: {e}")
            return None

    async def _save_compiled_model(self) -> None:
        """Save the compiled model to disk for reuse."""
        if not self.compiled_model:
            return
        
        try:
            model_path = Path("models") / f"pat_{self.pat_service.model_size}_optimized.pt"
            model_path.parent.mkdir(exist_ok=True)
            
            torch.jit.save(self.compiled_model, str(model_path))
            logger.info(f"Compiled model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save compiled model: {e}")

    def _generate_cache_key(self, input_data: ActigraphyInput) -> str:
        """Generate a cache key for actigraphy input."""
        # Create hash from user_id, data points, and parameters
        data_str = f"{input_data.user_id}_{len(input_data.data_points)}_{input_data.sampling_rate}"
        
        # Add hash of first/last few data points for uniqueness
        if input_data.data_points:
            first_vals = [p.value for p in input_data.data_points[:5]]
            last_vals = [p.value for p in input_data.data_points[-5:]]
            data_str += f"_{hash(tuple(first_vals + last_vals))}"
        
        return hashlib.md5(data_str.encode()).hexdigest()

    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - timestamp < self._cache_ttl

    async def optimized_analyze(
        self, input_data: ActigraphyInput, use_cache: bool = True
    ) -> tuple[ActigraphyAnalysis, bool]:
        """Optimized analysis with caching and model optimization.
        
        Returns:
            Tuple of (analysis_result, was_cached)
        """
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(input_data)
            if cache_key in self._cache:
                cached_result, timestamp = self._cache[cache_key]
                if self._is_cache_valid(timestamp):
                    logger.info(f"Cache hit for analysis {cache_key[:8]}")
                    return cached_result, True
                else:
                    # Remove expired entry
                    del self._cache[cache_key]

        # Run analysis with optimized model if available
        start_time = time.time()
        
        if self.optimization_enabled and self.compiled_model:
            result = await self._run_optimized_inference(input_data)
        else:
            result = await self.pat_service.analyze_actigraphy(input_data)
        
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.3f}s")
        
        # Cache the result
        if use_cache:
            cache_key = self._generate_cache_key(input_data)
            self._cache[cache_key] = (result, time.time())
            logger.info(f"Cached analysis result {cache_key[:8]}")
        
        return result, False

    async def _run_optimized_inference(
        self, input_data: ActigraphyInput
    ) -> ActigraphyAnalysis:
        """Run inference using the optimized compiled model."""
        if not self.compiled_model:
            # Fallback to regular inference
            return await self.pat_service.analyze_actigraphy(input_data)
        
        # Preprocess input data
        input_tensor = self.pat_service._preprocess_actigraphy_data(input_data.data_points)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        
        # Run optimized inference
        with torch.no_grad():
            outputs = self.compiled_model(input_tensor)
            
            # Convert ScriptModule output to dict format
            if isinstance(outputs, torch.Tensor):
                # Handle single tensor output (might need adjustment)
                batch_size = outputs.size(0)
                logits = outputs
                
                outputs_dict = {
                    "raw_logits": logits,
                    "sleep_metrics": torch.sigmoid(logits[:, :8]),
                    "circadian_score": torch.sigmoid(logits[:, 8:9]),
                    "depression_risk": torch.sigmoid(logits[:, 9:10]),
                    "embeddings": torch.randn(batch_size, 96),  # Placeholder
                }
            else:
                outputs_dict = outputs
        
        # Post-process results
        return self.pat_service._postprocess_predictions(outputs_dict, input_data.user_id)

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self._cache.clear()
        logger.info("Analysis cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        valid_entries = sum(
            1 for _, timestamp in self._cache.values() 
            if self._is_cache_valid(timestamp)
        )
        
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "cache_hit_ratio": self._calculate_hit_ratio(),
            "cache_ttl_seconds": self._cache_ttl,
        }

    def _calculate_hit_ratio(self) -> float:
        """Calculate cache hit ratio (placeholder implementation)."""
        # In a real implementation, you'd track hits/misses
        return 0.0

    async def warm_up(self, num_iterations: int = 5) -> dict[str, float]:
        """Warm up the optimized model with dummy data."""
        logger.info(f"Warming up PAT model with {num_iterations} iterations...")
        
        times = []
        for i in range(num_iterations):
            # Create dummy data
            from datetime import UTC, datetime, timedelta
            from clarity.ml.preprocessing import ActigraphyDataPoint
            
            dummy_points = [
                ActigraphyDataPoint(
                    timestamp=datetime.now(UTC) + timedelta(minutes=j),
                    value=float(j % 100)
                )
                for j in range(1440)
            ]
            
            dummy_data = ActigraphyInput(
                user_id=f"warmup_{i}",
                data_points=dummy_points,
                sampling_rate=1.0,
                duration_hours=24
            )
            
            start_time = time.time()
            await self.optimized_analyze(dummy_data, use_cache=False)
            iteration_time = time.time() - start_time
            times.append(iteration_time)
            
            logger.info(f"Warmup iteration {i+1}: {iteration_time:.3f}s")
        
        stats = {
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "total_time": sum(times)
        }
        
        logger.info(f"Warmup completed - Mean: {stats['mean_time']:.3f}s")
        return stats


class PATBatchProcessor:
    """Batch processor for handling multiple PAT analysis requests efficiently."""

    def __init__(self, optimizer: PATPerformanceOptimizer, max_batch_size: int = 8):
        self.optimizer = optimizer
        self.max_batch_size = max_batch_size
        self.pending_requests: list[tuple[ActigraphyInput, asyncio.Future]] = []
        self.processing = False

    async def submit_batch_analysis(
        self, input_data: ActigraphyInput
    ) -> ActigraphyAnalysis:
        """Submit analysis request for batch processing."""
        future: asyncio.Future[ActigraphyAnalysis] = asyncio.Future()
        
        self.pending_requests.append((input_data, future))
        
        # Trigger batch processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_batch())
        
        return await future

    async def _process_batch(self) -> None:
        """Process pending requests in batches."""
        if self.processing:
            return
        
        self.processing = True
        
        try:
            while self.pending_requests:
                # Take up to max_batch_size requests
                batch = self.pending_requests[:self.max_batch_size]
                self.pending_requests = self.pending_requests[self.max_batch_size:]
                
                # Process batch concurrently
                tasks = [
                    self.optimizer.optimized_analyze(input_data)
                    for input_data, _ in batch
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Return results to futures
                for (_, future), result in zip(batch, results):
                    if isinstance(result, Exception):
                        future.set_exception(result)
                    else:
                        analysis, _ = result  # Unpack (analysis, was_cached)
                        future.set_result(analysis)
                
                logger.info(f"Processed batch of {len(batch)} requests")
                
        finally:
            self.processing = False


@lru_cache(maxsize=1)
def get_pat_optimizer() -> PATPerformanceOptimizer:
    """Get the global PAT performance optimizer instance."""
    from clarity.ml.pat_service import get_pat_service
    
    # This would need to be called after the service is loaded
    # In practice, you'd initialize this during app startup
    raise NotImplementedError(
        "Call initialize_pat_optimizer() during app startup"
    )


async def initialize_pat_optimizer() -> PATPerformanceOptimizer:
    """Initialize the PAT performance optimizer during app startup."""
    pat_service = await get_pat_service()
    optimizer = PATPerformanceOptimizer(pat_service)
    
    # Apply default optimizations
    await optimizer.optimize_model(
        use_torchscript=True,
        use_pruning=False,  # Disable pruning by default to maintain accuracy
    )
    
    # Warm up the model
    await optimizer.warm_up(num_iterations=3)
    
    logger.info("PAT performance optimizer initialized successfully")
    return optimizer