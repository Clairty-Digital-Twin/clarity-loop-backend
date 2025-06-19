"""Local ML Model Development Server.

FastAPI-based local server for ML model development without AWS dependencies.
Provides REST API for model management, serving, and testing.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from clarity.ml.models.manager import LoadingStrategy, ModelLoadConfig, ModelManager
from clarity.ml.models.registry import (
    LEGACY_PAT_MODELS,
    ModelRegistry,
    ModelRegistryConfig,
    initialize_legacy_models,
)

logger = logging.getLogger(__name__)


class ModelServerConfig(BaseModel):
    """Configuration for local model server."""

    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8900
    log_level: str = "info"
    enable_cors: bool = True
    models_dir: Path = Path("./local_models")
    registry_file: Path = Path("./local_models/registry.json")
    auto_load_models: bool = True
    create_mock_models: bool = True
    mock_response_delay: float = 0.1  # Simulate inference latency


class PredictionRequest(BaseModel):
    """Request model for predictions."""

    model_id: str
    version: str = "latest"
    inputs: dict[str, Any]
    options: dict[str, Any] = {}


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    model_id: str
    version: str
    outputs: dict[str, Any]
    latency_ms: float
    model_info: dict[str, Any]


class ModelStatusResponse(BaseModel):
    """Response model for model status."""

    model_id: str
    version: str
    status: str
    metrics: dict[str, Any]
    metadata: dict[str, Any]


class MockPATModel:
    """Mock PAT model for local development."""

    def __init__(self, model_size: str = "medium") -> None:
        self.model_size = model_size
        self.config = {
            "small": {"features": 64, "layers": 4},
            "medium": {"features": 128, "layers": 8},
            "large": {"features": 256, "layers": 12},
        }.get(model_size, {"features": 128, "layers": 8})

    async def predict_async(self, data: dict[str, Any]) -> dict[str, Any]:
        """Mock prediction that simulates PAT model outputs."""
        await asyncio.sleep(0.05)  # Simulate processing time

        # Generate mock actigraphy analysis results
        return {
            "sleep_stages": [
                {"stage": "deep", "duration_minutes": 120, "confidence": 0.85},
                {"stage": "rem", "duration_minutes": 90, "confidence": 0.78},
                {"stage": "light", "duration_minutes": 180, "confidence": 0.92},
                {"stage": "wake", "duration_minutes": 30, "confidence": 0.95},
            ],
            "sleep_metrics": {
                "total_sleep_time": 420,  # minutes
                "sleep_efficiency": 0.87,
                "wake_episodes": 3,
                "sleep_onset_latency": 12,
                "rem_latency": 85,
            },
            "activity_patterns": {
                "activity_score": 0.65,
                "circadian_rhythm_strength": 0.78,
                "rest_activity_ratio": 0.23,
            },
            "model_info": {
                "model_size": self.model_size,
                "confidence_threshold": 0.7,
                "processing_time_ms": 50,
            },
        }


class LocalModelServer:
    """Local ML Model Development Server."""

    def __init__(self, config: ModelServerConfig | None = None) -> None:
        self.config = config or ModelServerConfig()
        self.app = FastAPI(
            title="Clarity ML Model Server",
            description="Local development server for ML models",
            version="1.0.0",
        )
        self.model_manager: ModelManager | None = None
        self.mock_models: dict[str, MockPATModel] = {}

        # Setup FastAPI routes
        self._setup_routes()

        # Setup CORS if enabled
        if self.config.enable_cors:
            self._setup_cors()

    async def startup_event(self) -> None:
        """Initialize model manager and load models."""
        try:
            # Ensure models directory exists
            self.config.models_dir.mkdir(parents=True, exist_ok=True)

            # Configure model registry for local development
            registry_config = ModelRegistryConfig(
                base_path=self.config.models_dir,
                cache_dir=self.config.models_dir / "cache",
                registry_file=self.config.registry_file,
                enable_local_server=True,
                local_server_port=self.config.port,
            )

            # Configure model loading for local development
            load_config = ModelLoadConfig(
                strategy=LoadingStrategy.LAZY,  # Load on demand for dev
                timeout_seconds=60,
                enable_monitoring=True,
            )

            # Initialize model manager
            registry = ModelRegistry(registry_config)
            self.model_manager = ModelManager(registry, registry_config, load_config)

            await self.model_manager.initialize()

            # Initialize legacy models if requested
            if self.config.auto_load_models:
                await initialize_legacy_models(registry)

            # Create mock models for local development
            if self.config.create_mock_models:
                await self._setup_mock_models()

            logger.info(
                "Local model server started on %s:%s",
                self.config.host,
                self.config.port,
            )

        except Exception as e:
            logger.exception("Failed to start model server: %s", e)
            raise

    async def _setup_mock_models(self) -> None:
        """Setup mock models for local development."""
        for model_id, metadata in LEGACY_PAT_MODELS.items():
            size_map = {"PAT-S": "small", "PAT-M": "medium", "PAT-L": "large"}
            size = size_map.get(model_id, "medium")

            self.mock_models[metadata.unique_id] = MockPATModel(size)
            logger.info("Created mock model: %s", metadata.unique_id)

    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""

        @self.app.on_event("startup")
        async def startup() -> None:
            await self.startup_event()

        @self.app.get("/")
        async def root() -> dict[str, Any]:  # noqa: RUF029
            return {
                "service": "Clarity ML Model Server",
                "version": "1.0.0",
                "status": "running",
                "models_loaded": (
                    len(self.model_manager.loaded_models) if self.model_manager else 0
                ),
            }

        @self.app.get("/health")
        async def health_check() -> dict[str, Any] | JSONResponse:
            """Health check endpoint."""
            if not self.model_manager:
                return JSONResponse({"status": "starting"}, status_code=503)

            return await self.model_manager.health_check()

        @self.app.get("/models", response_model=list[dict[str, Any]])
        async def list_models() -> list[dict[str, Any]]:
            """List all available models."""
            if not self.model_manager:
                raise HTTPException(
                    status_code=503, detail="Model manager not initialized"
                )

            models = await self.model_manager.registry.list_models()
            return [model.to_dict() for model in models]

        @self.app.get("/models/{model_id}", response_model=list[dict[str, Any]])
        async def get_model_versions(model_id: str) -> list[dict[str, Any]]:
            """Get all versions of a specific model."""
            if not self.model_manager:
                raise HTTPException(
                    status_code=503, detail="Model manager not initialized"
                )

            models = await self.model_manager.registry.list_models(model_id)
            if not models:
                raise HTTPException(
                    status_code=404, detail=f"Model {model_id} not found"
                )

            return [model.to_dict() for model in models]

        @self.app.get(
            "/models/{model_id}/{version}/status", response_model=ModelStatusResponse
        )
        async def get_model_status(
            model_id: str, version: str = "latest"
        ) -> ModelStatusResponse:
            """Get status of a specific model version."""
            if not self.model_manager:
                raise HTTPException(
                    status_code=503, detail="Model manager not initialized"
                )

            # Check if model is loaded
            loaded_model = await self.model_manager.get_model(model_id, version)
            if not loaded_model:
                # Check if model exists in registry
                metadata = await self.model_manager.registry.get_model(
                    model_id, version
                )
                if not metadata:
                    raise HTTPException(
                        status_code=404, detail=f"Model {model_id}:{version} not found"
                    )

                return ModelStatusResponse(
                    model_id=model_id,
                    version=version,
                    status="not_loaded",
                    metrics={},
                    metadata=metadata.to_dict(),
                )

            metrics = loaded_model.get_metrics()
            return ModelStatusResponse(
                model_id=model_id,
                version=version,
                status=loaded_model.status.value,
                metrics=metrics.dict(),
                metadata=loaded_model.metadata.to_dict(),
            )

        @self.app.post("/models/{model_id}/{version}/load")
        async def load_model(
            model_id: str,
            version: str = "latest",
            background_tasks: BackgroundTasks | None = None,
        ) -> dict[str, Any]:
            """Load a specific model version."""
            if not self.model_manager:
                raise HTTPException(
                    status_code=503, detail="Model manager not initialized"
                )

            # Start loading in background
            if background_tasks:
                background_tasks.add_task(
                    self.model_manager.preload_model, model_id, version
                )
                return {
                    "status": "loading_started",
                    "model_id": model_id,
                    "version": version,
                }
            success = await self.model_manager.preload_model(model_id, version)
            return {
                "status": "loaded" if success else "failed",
                "model_id": model_id,
                "version": version,
            }

        @self.app.post("/models/{model_id}/{version}/unload")
        async def unload_model(
            model_id: str, version: str = "latest"
        ) -> dict[str, Any]:
            """Unload a specific model version."""
            if not self.model_manager:
                raise HTTPException(
                    status_code=503, detail="Model manager not initialized"
                )

            success = await self.model_manager.unload_model(model_id, version)
            return {
                "status": "unloaded" if success else "not_loaded",
                "model_id": model_id,
                "version": version,
            }

        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest) -> PredictionResponse:
            """Make prediction using specified model."""
            if not self.model_manager:
                raise HTTPException(
                    status_code=503, detail="Model manager not initialized"
                )

            start_time = asyncio.get_event_loop().time()

            # Try to get loaded model first
            loaded_model = await self.model_manager.get_model(
                request.model_id, request.version
            )

            if loaded_model:
                # Use real model
                try:
                    outputs = await loaded_model.predict(**request.inputs)
                    latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000

                    return PredictionResponse(
                        model_id=request.model_id,
                        version=request.version,
                        outputs=outputs,
                        latency_ms=latency_ms,
                        model_info=loaded_model.metadata.to_dict(),
                    )
                except Exception as e:
                    logger.exception("Prediction failed: %s", e)
                    raise HTTPException(
                        status_code=500, detail=f"Prediction failed: {e!s}"
                    ) from e

            # Fallback to mock model for development
            unique_id = f"{request.model_id}:{request.version}"
            if unique_id in self.mock_models:
                await asyncio.sleep(self.config.mock_response_delay)
                mock_model = self.mock_models[unique_id]
                outputs = await mock_model.predict_async(request.inputs)
                latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000

                return PredictionResponse(
                    model_id=request.model_id,
                    version=request.version,
                    outputs=outputs,
                    latency_ms=latency_ms,
                    model_info={"mock": True, "model_size": mock_model.model_size},
                )

            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_id}:{request.version} not available",
            )

        @self.app.get("/metrics")
        async def get_metrics() -> dict[str, Any]:
            """Get performance metrics for all models."""
            if not self.model_manager:
                raise HTTPException(
                    status_code=503, detail="Model manager not initialized"
                )

            return await self.model_manager.get_all_metrics()

        @self.app.get("/download-progress/{model_id}")
        async def get_download_progress(
            model_id: str, version: str = Query(default="latest")
        ) -> dict[str, Any]:
            """Get download progress for a model."""
            if not self.model_manager:
                raise HTTPException(
                    status_code=503, detail="Model manager not initialized"
                )

            progress = await self.model_manager.registry.get_download_progress(
                model_id, version
            )
            if not progress:
                raise HTTPException(status_code=404, detail="No download in progress")

            return progress

        @self.app.post("/create-mock-data")
        async def create_mock_data() -> dict[str, Any]:  # noqa: RUF029
            """Create mock training data for testing."""
            import random  # noqa: PLC0415

            # Generate mock actigraphy data
            return {
                "actigraphy_data": {
                    "timestamps": [f"2024-01-01T{i:02d}:00:00Z" for i in range(24)],
                    "activity_counts": [
                        random.randint(0, 1000) for _ in range(24)  # noqa: S311
                    ],
                    "light_levels": [
                        random.randint(0, 10000) for _ in range(24)  # noqa: S311
                    ],
                },
                "metadata": {
                    "subject_id": "test_001",
                    "device_type": "ActiGraph_GT3X",
                    "sampling_rate": "1_minute",
                    "duration_hours": 24,
                },
            }

    def _setup_cors(self) -> None:
        """Setup CORS for frontend development."""
        from fastapi.middleware.cors import CORSMiddleware  # noqa: PLC0415

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, specify actual origins
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def run(self) -> None:
        """Run the local model server."""
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level=self.config.log_level,
        )


# CLI interface for local model server
async def create_placeholder_models(models_dir: Path) -> None:  # noqa: RUF029
    """Create placeholder models for local development."""
    models_dir.mkdir(parents=True, exist_ok=True)

    for model_id in LEGACY_PAT_MODELS:
        model_file = models_dir / f"{model_id}.h5"

        if not model_file.exists():
            # Create a small placeholder file
            with model_file.open("wb") as f:
                f.write(b"PLACEHOLDER_MODEL_DATA" * 1000)  # ~20KB file

            logger.info("Created placeholder model: %s", model_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clarity Local Model Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")  # noqa: S104
    parser.add_argument("--port", type=int, default=8900, help="Server port")
    parser.add_argument(
        "--models-dir", type=Path, default="./local_models", help="Models directory"
    )
    parser.add_argument(
        "--create-placeholders", action="store_true", help="Create placeholder models"
    )
    parser.add_argument("--log-level", default="info", help="Log level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create placeholder models if requested
    if args.create_placeholders:
        asyncio.run(create_placeholder_models(args.models_dir))

    # Create and run server
    config = ModelServerConfig(
        host=args.host,
        port=args.port,
        models_dir=args.models_dir,
        log_level=args.log_level,
    )

    server = LocalModelServer(config)
    server.run()
