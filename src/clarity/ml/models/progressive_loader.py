"""Progressive ML Model Loading Service.

This service handles intelligent model loading for the main application,
replacing the legacy S3 download approach with progressive loading strategies.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import StrEnum
import logging
import os
from pathlib import Path
import time
from typing import Any

from pydantic import BaseModel

from clarity.ml.models.manager import (
    LoadedModel,
    LoadingStrategy,
    ModelLoadConfig,
    ModelManager,
)
from clarity.ml.models.registry import (
    ModelRegistry,
    ModelRegistryConfig,
    initialize_legacy_models,
)

logger = logging.getLogger(__name__)


class ApplicationPhase(StrEnum):
    """Application startup phases."""

    INIT = "init"  # Basic initialization
    CORE_READY = "core_ready"  # Core services ready
    MODELS_LOADING = "models_loading"  # Models loading in progress
    MODELS_READY = "models_ready"  # Critical models loaded
    FULLY_READY = "fully_ready"  # All models loaded


class ProgressiveLoadingConfig(BaseModel):
    """Configuration for progressive loading service."""

    # Environment detection
    is_production: bool = True
    is_local_dev: bool = False
    enable_efs_cache: bool = True

    # Loading strategy
    loading_strategy: LoadingStrategy = LoadingStrategy.PROGRESSIVE
    critical_models: list[str] = ["pat:latest"]
    preload_models: list[str] = ["pat:stable", "pat:fast"]

    # Performance settings
    max_parallel_loads: int = 2
    startup_timeout_seconds: int = 300
    model_load_timeout_seconds: int = 180

    # Fallback settings
    enable_graceful_degradation: bool = True
    fallback_to_mock: bool = False  # Only for local dev

    # Monitoring
    enable_metrics: bool = True
    metrics_interval_seconds: int = 30


class ModelAvailabilityStatus(BaseModel):
    """Status of model availability."""

    model_id: str
    version: str
    status: str  # "available", "loading", "failed", "not_found"
    load_time_seconds: float | None = None
    error_message: str | None = None
    is_critical: bool = False


class ProgressiveLoadingService:
    """Progressive ML Model Loading Service.

    This service provides intelligent model loading with the following features:
    - Progressive loading: Start app quickly, load models in background
    - Critical path optimization: Load essential models first
    - Graceful degradation: Continue serving with available models
    - EFS caching in production: Shared model storage across containers
    - Local development support: Mock models and local server
    - Performance monitoring: Track loading times and availability
    """

    def __init__(self, config: ProgressiveLoadingConfig | None = None) -> None:
        self.config = config or ProgressiveLoadingConfig()
        self.model_manager: ModelManager | None = None
        self.current_phase = ApplicationPhase.INIT
        self.model_status: dict[str, ModelAvailabilityStatus] = {}
        self.loading_tasks: dict[str, asyncio.Task[Any]] = {}
        self.startup_time = time.time()
        self.phase_transitions: dict[ApplicationPhase, float] = {}
        self._lock = asyncio.Lock()

        # Auto-detect environment
        self._detect_environment()

        logger.info(
            "Progressive loading service initialized for %s",
            "production" if self.config.is_production else "development",
        )

    async def initialize(self) -> bool:
        """Initialize the progressive loading service."""
        try:
            self.current_phase = ApplicationPhase.INIT

            # Setup model registry and manager
            await self._setup_model_infrastructure()

            # Register legacy models if needed
            await self._initialize_model_registry()

            self.current_phase = ApplicationPhase.CORE_READY
            self._record_phase_transition(ApplicationPhase.CORE_READY)

            # Start progressive loading
            if self.config.loading_strategy == LoadingStrategy.PROGRESSIVE:
                await self._start_progressive_loading()
            elif self.config.loading_strategy == LoadingStrategy.EAGER:
                await self._start_eager_loading()

            logger.info("Progressive loading service initialized successfully")
            return True

        except Exception as e:
            logger.exception("Failed to initialize progressive loading service: %s", e)
            return False

    async def get_model(
        self,
        model_id: str,
        version: str = "latest",
        timeout: int | None = None,
        *,
        critical: bool = False,
    ) -> LoadedModel | None:
        """Get a model, loading it if necessary.

        Args:
            model_id: Model identifier
            version: Model version (default: latest)
            timeout: Maximum wait time for loading
            critical: Whether this is a critical model request
        """
        if not self.model_manager:
            logger.error("Model manager not initialized")
            return None

        unique_id = f"{model_id}:{version}"

        # Update status tracking
        if unique_id not in self.model_status:
            self.model_status[unique_id] = ModelAvailabilityStatus(
                model_id=model_id,
                version=version,
                status="not_found",
                is_critical=critical,
            )

        # Try to get the model
        try:
            model = await self.model_manager.get_model(model_id, version, timeout)

            if model:
                self.model_status[unique_id].status = "available"
                if self.model_status[unique_id].load_time_seconds is None:
                    self.model_status[unique_id].load_time_seconds = (
                        time.time() - self.startup_time
                    )
                return model
            self.model_status[unique_id].status = "failed"
            return None

        except Exception as e:
            logger.exception("Failed to get model %s: %s", unique_id, e)
            self.model_status[unique_id].status = "failed"
            self.model_status[unique_id].error_message = str(e)
            return None

    async def preload_model(self, model_id: str, version: str = "latest") -> bool:
        """Preload a model in the background."""
        if not self.model_manager:
            return False

        return await self.model_manager.preload_model(model_id, version)

    async def get_application_status(self) -> dict[str, Any]:
        """Get comprehensive application and model status."""
        uptime_seconds = time.time() - self.startup_time

        # Count models by status
        status_counts: dict[str, int] = {}
        critical_ready = 0
        critical_total = 0

        for model_status in self.model_status.values():
            status = model_status.status
            status_counts[status] = status_counts.get(status, 0) + 1

            if model_status.is_critical:
                critical_total += 1
                if status == "available":
                    critical_ready += 1

        # Determine overall application status
        overall_status = "healthy"
        if self.current_phase in {ApplicationPhase.INIT, ApplicationPhase.CORE_READY}:
            overall_status = "starting"
        elif critical_ready < critical_total:
            overall_status = "degraded"
        elif status_counts.get("failed", 0) > 0:
            overall_status = "partial"

        return {
            "overall_status": overall_status,
            "current_phase": self.current_phase.value,
            "uptime_seconds": uptime_seconds,
            "models": {
                "total": len(self.model_status),
                "available": status_counts.get("available", 0),
                "loading": status_counts.get("loading", 0),
                "failed": status_counts.get("failed", 0),
                "critical_ready": f"{critical_ready}/{critical_total}",
            },
            "phase_transitions": self.phase_transitions,
            "environment": {
                "is_production": self.config.is_production,
                "is_local_dev": self.config.is_local_dev,
                "efs_enabled": self.config.enable_efs_cache,
            },
        }

    async def get_model_status(
        self, model_id: str, version: str = "latest"
    ) -> ModelAvailabilityStatus | None:
        """Get status of a specific model."""
        unique_id = f"{model_id}:{version}"
        return self.model_status.get(unique_id)

    async def get_all_model_status(self) -> list[ModelAvailabilityStatus]:
        """Get status of all tracked models."""
        return list(self.model_status.values())

    async def wait_for_critical_models(self, timeout: int = 300) -> bool:
        """Wait for critical models to be ready."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            critical_ready = True

            for model_spec in self.config.critical_models:
                model_id, version = self._parse_model_spec(model_spec)
                unique_id = f"{model_id}:{version}"

                status = self.model_status.get(unique_id)
                if not status or status.status != "available":
                    critical_ready = False
                    break

            if critical_ready:
                self.current_phase = ApplicationPhase.MODELS_READY
                self._record_phase_transition(ApplicationPhase.MODELS_READY)
                return True

            await asyncio.sleep(1)

        logger.warning("Critical models not ready within %s seconds", timeout)
        return False

    async def health_check(self) -> dict[str, Any]:
        """Comprehensive health check."""
        if not self.model_manager:
            return {"status": "unhealthy", "reason": "model_manager_not_initialized"}

        try:
            manager_health = await self.model_manager.health_check()
            app_status = await self.get_application_status()

            # Determine health status
            is_healthy = (
                app_status["overall_status"] in {"healthy", "partial"}
                and manager_health["status"] == "healthy"
            )

            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "progressive_loading": app_status,
                "model_manager": manager_health,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.exception("Health check failed: %s", e)
            return {"status": "unhealthy", "error": str(e)}

    def _detect_environment(self) -> None:
        """Auto-detect deployment environment."""
        # Check for common production environment indicators
        is_prod_env = any(
            [
                os.getenv("ENVIRONMENT") == "production",
                os.getenv("AWS_EXECUTION_ENV"),  # AWS Lambda/ECS
                os.getenv("ECS_CONTAINER_METADATA_URI"),  # ECS
                Path("/app").exists(),  # Container environment
            ]
        )

        # Check for local development indicators
        is_local_dev = any(
            [
                os.getenv("ENVIRONMENT") == "development",
                os.getenv("FLASK_ENV") == "development",
                os.getenv("FASTAPI_ENV") == "development",
                not is_prod_env and Path("./src").exists(),
            ]
        )

        self.config.is_production = is_prod_env
        self.config.is_local_dev = is_local_dev

        # Adjust config based on environment
        if is_local_dev:
            self.config.fallback_to_mock = True
            self.config.enable_efs_cache = False

        logger.info(
            "Environment detected: production=%s, local_dev=%s",
            is_prod_env,
            is_local_dev,
        )

    async def _setup_model_infrastructure(self) -> None:
        """Setup model registry and manager."""
        # Configure paths based on environment
        if self.config.is_production and self.config.enable_efs_cache:
            # Use EFS mount in production
            base_path = os.getenv("EFS_MODELS_PATH", "/mnt/efs/models")
            cache_dir = os.getenv("EFS_CACHE_PATH", "/mnt/efs/cache")
        else:
            # Use local paths
            base_path = "./models"
            cache_dir = "./cache/models"

        registry_config = ModelRegistryConfig(
            base_path=base_path,
            cache_dir=cache_dir,
            enable_local_server=self.config.is_local_dev,
        )

        load_config = ModelLoadConfig(
            strategy=self.config.loading_strategy,
            timeout_seconds=self.config.model_load_timeout_seconds,
            enable_monitoring=self.config.enable_metrics,
        )

        # Create model manager
        registry = ModelRegistry(registry_config)
        self.model_manager = ModelManager(registry, registry_config, load_config)

        await self.model_manager.initialize()

    async def _initialize_model_registry(self) -> None:
        """Initialize model registry with legacy models."""
        if self.model_manager:
            await initialize_legacy_models(self.model_manager.registry)

    async def _start_progressive_loading(self) -> None:
        """Start progressive model loading."""
        self.current_phase = ApplicationPhase.MODELS_LOADING
        self._record_phase_transition(ApplicationPhase.MODELS_LOADING)

        # Create semaphore for controlling parallel loads
        semaphore = asyncio.Semaphore(self.config.max_parallel_loads)

        async def load_model_with_semaphore(
            model_spec: str, *, is_critical: bool = False
        ) -> None:
            async with semaphore:
                model_id, version = self._parse_model_spec(model_spec)
                unique_id = f"{model_id}:{version}"

                # Update status
                self.model_status[unique_id] = ModelAvailabilityStatus(
                    model_id=model_id,
                    version=version,
                    status="loading",
                    is_critical=is_critical,
                )

                start_time = time.time()
                try:
                    success = (
                        await self.model_manager.preload_model(model_id, version)
                        if self.model_manager
                        else False
                    )
                    load_time = time.time() - start_time

                    if success:
                        self.model_status[unique_id].status = "available"
                        self.model_status[unique_id].load_time_seconds = load_time
                        logger.info("Loaded model %s in %.2fs", unique_id, load_time)
                    else:
                        self.model_status[unique_id].status = "failed"
                        logger.error("Failed to load model %s", unique_id)

                except Exception as e:
                    self.model_status[unique_id].status = "failed"
                    self.model_status[unique_id].error_message = str(e)
                    logger.exception("Error loading model %s: %s", unique_id, e)

        # Start loading critical models first
        critical_tasks = []
        for model_spec in self.config.critical_models:
            task = asyncio.create_task(
                load_model_with_semaphore(model_spec, is_critical=True)
            )
            critical_tasks.append(task)

        # Start loading preload models
        preload_tasks = []
        for model_spec in self.config.preload_models:
            if model_spec not in self.config.critical_models:  # Avoid duplicates
                task = asyncio.create_task(
                    load_model_with_semaphore(model_spec, is_critical=False)
                )
                preload_tasks.append(task)

        # Wait for critical models with timeout
        if critical_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*critical_tasks, return_exceptions=True),
                    timeout=self.config.startup_timeout_seconds,
                )
                self.current_phase = ApplicationPhase.MODELS_READY
                self._record_phase_transition(ApplicationPhase.MODELS_READY)
            except TimeoutError:
                logger.warning("Critical models loading timed out")

        # Continue preload tasks in background
        if preload_tasks:
            self._preload_task = asyncio.create_task(
                self._finish_preloading(preload_tasks)
            )

    async def _finish_preloading(self, preload_tasks: list[asyncio.Task[Any]]) -> None:
        """Finish preloading non-critical models in background."""
        try:
            await asyncio.gather(*preload_tasks, return_exceptions=True)
            self.current_phase = ApplicationPhase.FULLY_READY
            self._record_phase_transition(ApplicationPhase.FULLY_READY)
            logger.info("All model preloading completed")
        except Exception as e:
            logger.exception("Error in background preloading: %s", e)

    async def _start_eager_loading(self) -> None:
        """Start eager loading of all models."""
        self.current_phase = ApplicationPhase.MODELS_LOADING
        self._record_phase_transition(ApplicationPhase.MODELS_LOADING)

        # Load all registered models
        if not self.model_manager:
            return
        all_models = await self.model_manager.registry.list_models()

        load_tasks = []
        for metadata in all_models:
            unique_id = f"{metadata.model_id}:{metadata.version}"
            self.model_status[unique_id] = ModelAvailabilityStatus(
                model_id=metadata.model_id,
                version=metadata.version,
                status="loading",
                is_critical=True,  # All models are critical in eager mode
            )

            task = asyncio.create_task(
                self.model_manager.preload_model(metadata.model_id, metadata.version)
            )
            load_tasks.append((task, unique_id))

        # Wait for all models to load
        results = await asyncio.gather(
            *[task for task, _metadata in load_tasks], return_exceptions=True
        )

        # Update status
        for (_task, unique_id), result in zip(load_tasks, results, strict=False):
            if isinstance(result, Exception):
                self.model_status[unique_id].status = "failed"
                self.model_status[unique_id].error_message = str(result)
            elif result:
                self.model_status[unique_id].status = "available"
            else:
                self.model_status[unique_id].status = "failed"

        self.current_phase = ApplicationPhase.FULLY_READY
        self._record_phase_transition(ApplicationPhase.FULLY_READY)

    def _parse_model_spec(self, model_spec: str) -> tuple[str, str]:
        """Parse model specification string (e.g., 'pat:latest' -> ('pat', 'latest'))."""
        if ":" in model_spec:
            parts = model_spec.split(":", 1)
            return parts[0], parts[1]
        return model_spec, "latest"

    def _record_phase_transition(self, phase: ApplicationPhase) -> None:
        """Record phase transition timing."""
        self.phase_transitions[phase] = time.time() - self.startup_time
        logger.info(
            "Application phase: %s (%.2fs)", phase.value, self.phase_transitions[phase]
        )


# Global progressive loading service instance
_progressive_service: ProgressiveLoadingService | None = None


async def get_progressive_service(
    config: ProgressiveLoadingConfig | None = None,
) -> ProgressiveLoadingService:
    """Get or create the global progressive loading service."""
    global _progressive_service  # noqa: PLW0603 - Singleton pattern

    if _progressive_service is None:
        _progressive_service = ProgressiveLoadingService(config)
        await _progressive_service.initialize()

    return _progressive_service


@asynccontextmanager
async def progressive_loading_lifespan(
    config: ProgressiveLoadingConfig | None = None,
) -> AsyncGenerator[ProgressiveLoadingService, None]:
    """Context manager for progressive loading service lifecycle."""
    service = await get_progressive_service(config)
    try:
        yield service
    finally:
        # Cleanup if needed
        pass
