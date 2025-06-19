"""Revolutionary ML Model Manager.

High-level model management with progressive loading, caching, and monitoring.
This replaces the legacy monolithic PAT service approach.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
import logging
from pathlib import Path
import time
from typing import Any

from pydantic import BaseModel
import torch

from clarity.ml.models.registry import (
    ModelMetadata,
    ModelRegistry,
    ModelRegistryConfig,
    ModelStatus,
)
from clarity.ml.pat_service import PATModelService  # Legacy service for compatibility

logger = logging.getLogger(__name__)


class LoadingStrategy(StrEnum):
    """Model loading strategies."""

    EAGER = "eager"  # Load immediately on startup
    LAZY = "lazy"  # Load on first request
    PROGRESSIVE = "progressive"  # Load in background with fallback
    ON_DEMAND = "on_demand"  # Load only when specifically requested


@dataclass
class ModelLoadConfig:
    """Configuration for model loading behavior."""

    strategy: LoadingStrategy = LoadingStrategy.PROGRESSIVE
    timeout_seconds: int = 300
    max_memory_mb: int = 2048
    enable_quantization: bool = False
    warm_up_samples: int = 5
    fallback_to_cpu: bool = True
    enable_monitoring: bool = True


class ModelPerformanceMetrics(BaseModel):
    """Real-time model performance tracking."""

    model_id: str
    version: str
    total_inferences: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    error_count: int = 0
    last_used: float | None = None
    memory_usage_mb: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


class LoadedModel:
    """Wrapper for loaded model with metadata and performance tracking."""

    def __init__(
        self, metadata: ModelMetadata, model_instance: Any, config: ModelLoadConfig
    ) -> None:
        self.metadata = metadata
        self.model_instance = model_instance
        self.config = config
        self.status = ModelStatus.AVAILABLE
        self.load_time = time.time()
        self.metrics = ModelPerformanceMetrics(
            model_id=metadata.model_id, version=metadata.version
        )
        self._lock = asyncio.Lock()

    async def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Execute prediction with performance tracking."""
        start_time = time.time()

        async with self._lock:
            self.metrics.last_used = start_time

            try:
                # Execute the actual prediction
                if hasattr(self.model_instance, "predict_async"):
                    result = await self.model_instance.predict_async(*args, **kwargs)
                elif asyncio.iscoroutinefunction(self.model_instance.predict):
                    result = await self.model_instance.predict(*args, **kwargs)
                else:
                    # Run synchronous prediction in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, self.model_instance.predict, *args, **kwargs
                    )

                # Update metrics
                latency_ms = (time.time() - start_time) * 1000
                self.metrics.total_inferences += 1
                self.metrics.total_latency_ms += latency_ms
                self.metrics.avg_latency_ms = (
                    self.metrics.total_latency_ms / self.metrics.total_inferences
                )

                if self.config.enable_monitoring:
                    logger.debug(
                        "Inference completed for %s: %.2fms",
                        self.metadata.unique_id,
                        latency_ms,
                    )

                return result

            except Exception as e:
                self.metrics.error_count += 1
                logger.exception(
                    "Prediction failed for %s: %s", self.metadata.unique_id, e
                )
                raise

    def get_metrics(self) -> ModelPerformanceMetrics:
        """Get current performance metrics."""
        return self.metrics

    def update_memory_usage(self) -> None:
        """Update memory usage statistics."""
        if torch.cuda.is_available():
            self.metrics.memory_usage_mb = torch.cuda.memory_allocated() / (1024 * 1024)


class ModelManager:
    """Revolutionary ML Model Manager.

    Features:
    - Progressive model loading with fallback strategies
    - Intelligent caching and memory management
    - Real-time performance monitoring
    - Hot-swappable model versions
    - Local development mode support
    - A/B testing capabilities
    """

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        config: ModelRegistryConfig | None = None,
        load_config: ModelLoadConfig | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry(config or ModelRegistryConfig())
        self.load_config = load_config or ModelLoadConfig()
        self.loaded_models: dict[str, LoadedModel] = {}
        self.loading_tasks: dict[str, asyncio.Task[LoadedModel | None]] = {}
        self.model_factories: dict[str, Callable[[ModelMetadata], Any]] = {}
        self._lock = asyncio.Lock()

        # Register default model factories
        self._register_default_factories()

        logger.info("ModelManager initialized with progressive loading enabled")

    async def initialize(self) -> None:
        """Initialize model manager and start progressive loading."""
        await self.registry.initialize()

        if self.load_config.strategy == LoadingStrategy.EAGER:
            await self._load_all_models()
        elif self.load_config.strategy == LoadingStrategy.PROGRESSIVE:
            await self._start_progressive_loading()

        logger.info(
            "ModelManager initialized with %d models loaded", len(self.loaded_models)
        )

    async def get_model(
        self, model_id: str, version: str = "latest", timeout: int | None = None
    ) -> LoadedModel | None:
        """Get loaded model, loading it if necessary."""
        unique_id = f"{model_id}:{version if version != 'latest' else await self._resolve_latest_version(model_id)}"

        # Check if already loaded
        if unique_id in self.loaded_models:
            loaded_model = self.loaded_models[unique_id]
            loaded_model.metrics.cache_hits += 1
            return loaded_model

        # Check if currently loading
        if unique_id in self.loading_tasks:
            try:
                timeout_val = timeout or self.load_config.timeout_seconds
                task = self.loading_tasks[unique_id]
                return await asyncio.wait_for(task, timeout=timeout_val)
            except TimeoutError:
                logger.exception("Model loading timeout for %s", unique_id)
                return None

        # Start loading
        loaded_model_or_none = await self._load_model(model_id, version)
        if loaded_model_or_none:
            loaded_model_or_none.metrics.cache_misses += 1

        return loaded_model_or_none

    async def preload_model(self, model_id: str, version: str = "latest") -> bool:
        """Preload model in background."""
        try:
            model = await self.get_model(model_id, version)
            return model is not None
        except Exception as e:
            logger.exception("Failed to preload model %s:%s: %s", model_id, version, e)
            return False

    async def unload_model(self, model_id: str, version: str = "latest") -> bool:
        """Unload model from memory."""
        unique_id = f"{model_id}:{version if version != 'latest' else await self._resolve_latest_version(model_id)}"

        async with self._lock:
            if unique_id in self.loaded_models:
                self.loaded_models.pop(unique_id)

                # Clean up GPU memory if using CUDA
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info("Unloaded model %s", unique_id)
                return True

            return False

    async def swap_model_version(
        self, model_id: str, old_version: str, new_version: str
    ) -> bool:
        """Hot-swap model version for A/B testing."""
        try:
            # Load new version
            new_model = await self.get_model(model_id, new_version)
            if not new_model:
                logger.error(
                    "Failed to load new model version %s:%s", model_id, new_version
                )
                return False

            # Unload old version
            await self.unload_model(model_id, old_version)

            logger.info(
                "Successfully swapped %s from %s to %s",
                model_id,
                old_version,
                new_version,
            )
            return True

        except Exception as e:
            logger.exception("Model version swap failed: %s", e)
            return False

    async def get_all_metrics(self) -> dict[str, ModelPerformanceMetrics]:
        """Get performance metrics for all loaded models."""
        metrics = {}
        for unique_id, loaded_model in self.loaded_models.items():
            loaded_model.update_memory_usage()
            metrics[unique_id] = loaded_model.get_metrics()
        return metrics

    async def health_check(self) -> dict[str, Any]:
        """Comprehensive health check for all models."""
        health_status = {
            "status": "healthy",
            "loaded_models": len(self.loaded_models),
            "loading_tasks": len(self.loading_tasks),
            "total_memory_mb": 0.0,
            "models": {},
        }

        for unique_id, loaded_model in self.loaded_models.items():
            loaded_model.update_memory_usage()
            metrics = loaded_model.get_metrics()

            model_health = {
                "status": loaded_model.status.value,
                "uptime_seconds": time.time() - loaded_model.load_time,
                "total_inferences": metrics.total_inferences,
                "avg_latency_ms": metrics.avg_latency_ms,
                "error_rate": metrics.error_count / max(metrics.total_inferences, 1),
                "memory_usage_mb": metrics.memory_usage_mb,
            }

            models = health_status["models"]
            if not isinstance(models, dict):
                msg = "health_status['models'] must be a dict"
                raise TypeError(msg)
            models[unique_id] = model_health
            total_mem = health_status["total_memory_mb"]
            if not isinstance(total_mem, (int, float)):
                msg = "health_status['total_memory_mb'] must be numeric"
                raise TypeError(msg)
            health_status["total_memory_mb"] = total_mem + metrics.memory_usage_mb

        return health_status

    def register_model_factory(
        self, model_type: str, factory: Callable[[ModelMetadata], Any]
    ) -> None:
        """Register a factory function for creating model instances."""
        self.model_factories[model_type] = factory
        logger.info("Registered model factory for type: %s", model_type)

    async def _load_model(self, model_id: str, version: str) -> LoadedModel | None:
        """Load a specific model version."""
        unique_id = f"{model_id}:{version if version != 'latest' else await self._resolve_latest_version(model_id)}"

        if unique_id in self.loading_tasks:
            return await self.loading_tasks[unique_id]

        # Create loading task
        self.loading_tasks[unique_id] = asyncio.create_task(
            self._load_model_task(model_id, version)
        )

        try:
            loaded_model = await self.loading_tasks[unique_id]
            if loaded_model:
                async with self._lock:
                    self.loaded_models[unique_id] = loaded_model
            return loaded_model
        finally:
            self.loading_tasks.pop(unique_id, None)

    async def _load_model_task(self, model_id: str, version: str) -> LoadedModel | None:
        """Actual model loading task."""
        start_time = time.time()

        try:
            # Get model metadata
            metadata = await self.registry.get_model(model_id, version)
            if not metadata:
                logger.error("Model metadata not found: %s:%s", model_id, version)
                return None

            logger.info("Loading model %s...", metadata.unique_id)

            # Download model if not cached
            if not metadata.local_path or not Path(metadata.local_path).exists():
                logger.info("Downloading model %s...", metadata.unique_id)
                success = await self.registry.download_model(model_id, version)
                if not success:
                    logger.error("Failed to download model %s", metadata.unique_id)
                    return None

                # Refresh metadata to get updated local path
                metadata = await self.registry.get_model(model_id, version)
                if not metadata:
                    logger.error(
                        "Failed to refresh metadata for %s:%s", model_id, version
                    )
                    return None

            # Create model instance using appropriate factory
            model_instance = await self._create_model_instance(metadata)
            if not model_instance:
                logger.error(
                    "Failed to create model instance for %s", metadata.unique_id
                )
                return None

            # Create loaded model wrapper
            loaded_model = LoadedModel(metadata, model_instance, self.load_config)

            # Warm up model if configured
            if self.load_config.warm_up_samples > 0:
                await self._warm_up_model(loaded_model)

            load_time = time.time() - start_time
            logger.info(
                "Successfully loaded model %s in %.2fs", metadata.unique_id, load_time
            )

            return loaded_model

        except Exception as e:
            logger.exception("Failed to load model %s:%s: %s", model_id, version, e)
            return None

    async def _create_model_instance(self, metadata: ModelMetadata) -> Any:
        """Create model instance using registered factories."""
        # Determine model type from metadata or model_id
        model_type = metadata.model_id

        if model_type not in self.model_factories:
            logger.error("No factory registered for model type: %s", model_type)
            return None

        try:
            factory = self.model_factories[model_type]

            # Call factory with model metadata
            if asyncio.iscoroutinefunction(factory):
                return await factory(metadata)
            return factory(metadata)

        except Exception as e:
            logger.exception("Model factory failed for %s: %s", metadata.unique_id, e)
            return None

    async def _warm_up_model(self, loaded_model: LoadedModel) -> None:
        """Warm up model with dummy predictions."""
        try:
            logger.info("Warming up model %s...", loaded_model.metadata.unique_id)

            # This would be model-specific warm-up logic
            # For now, just update the timestamp
            loaded_model.metrics.last_used = time.time()

            logger.info(
                "Model warm-up completed for %s", loaded_model.metadata.unique_id
            )

        except (ValueError, TypeError, RuntimeError) as e:
            logger.warning(
                "Model warm-up failed for %s: %s", loaded_model.metadata.unique_id, e
            )

    async def _resolve_latest_version(self, model_id: str) -> str:
        """Resolve 'latest' alias to actual version."""
        # Check if 'latest' alias exists
        latest_model = await self.registry.get_model(model_id, "latest")
        if latest_model:
            return latest_model.version

        # Fallback: find highest version number
        models = await self.registry.list_models(model_id)
        if not models:
            return "latest"  # Return as-is if no models found

        # Sort by version (simple string sort for now, could be semantic versioning)
        latest = max(models, key=lambda m: m.version)
        return latest.version

    async def _load_all_models(self) -> None:
        """Load all registered models (eager loading)."""
        all_models = await self.registry.list_models()
        load_tasks = []

        for metadata in all_models:
            task = self._load_model(metadata.model_id, metadata.version)
            load_tasks.append(task)

        # Load all models concurrently
        results = await asyncio.gather(*load_tasks, return_exceptions=True)

        successful_loads = sum(1 for r in results if isinstance(r, LoadedModel))
        logger.info(
            "Eager loading completed: %d/%d models loaded",
            successful_loads,
            len(all_models),
        )

    async def _start_progressive_loading(self) -> None:
        """Start progressive loading in background."""
        self._loading_task = asyncio.create_task(self._progressive_loading_task())

    async def _progressive_loading_task(self) -> None:
        """Background task for progressive model loading."""
        try:
            # Load critical models first (based on aliases)
            priority_aliases = ["latest", "stable", "fast"]

            for alias in priority_aliases:
                try:
                    # Find models that match this alias
                    if alias in self.registry.aliases:
                        alias_obj = self.registry.aliases[alias]
                        await self._load_model(alias_obj.model_id, alias_obj.version)
                        await asyncio.sleep(1)  # Brief pause between loads
                except (ValueError, TypeError, RuntimeError, KeyError) as e:
                    logger.warning(
                        "Failed to load priority model with alias %s: %s", alias, e
                    )

            logger.info("Progressive loading of priority models completed")

        except Exception as e:
            logger.exception("Progressive loading task failed: %s", e)

    def _register_default_factories(self) -> None:
        """Register default model factories."""

        async def pat_model_factory(metadata: ModelMetadata) -> Any:
            """Factory for PAT models using legacy service."""
            try:
                # Determine model size from metadata
                model_size = "small"  # default
                if "medium" in metadata.name.lower() or metadata.tier.value == "medium":
                    model_size = "medium"
                elif "large" in metadata.name.lower() or metadata.tier.value == "large":
                    model_size = "large"

                # Create PAT service instance
                pat_service = PATModelService(model_size=model_size)

                # Override model path to use our downloaded model
                if metadata.local_path:
                    # Update the config to use our cached model
                    pat_service.config["model_path"] = metadata.local_path

                # Load the model asynchronously
                await pat_service.load_model()

                return pat_service

            except Exception as e:
                logger.exception("PAT model factory failed: %s", e)
                raise

        # Register PAT model factory
        self.register_model_factory("pat", pat_model_factory)


@asynccontextmanager
async def get_model_manager(
    config: ModelRegistryConfig | None = None,
    load_config: ModelLoadConfig | None = None,
) -> AsyncGenerator[ModelManager, None]:
    """Context manager for model manager lifecycle."""
    manager = ModelManager(config=config, load_config=load_config)

    try:
        await manager.initialize()
        yield manager
    finally:
        # Cleanup resources
        for unique_id in list(manager.loaded_models.keys()):
            await manager.unload_model(*unique_id.split(":", 1))
        logger.info("ModelManager cleanup completed")
