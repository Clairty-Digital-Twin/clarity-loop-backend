"""Revolutionary ML Model Management System.

This package provides a comprehensive ML model management solution with:
- Intelligent model registry with versioning
- Progressive loading with caching
- Local development server
- Performance monitoring
- CLI management tools
"""

from clarity.ml.models.local_server import (
    LocalModelServer,
    MockPATModel,
    ModelServerConfig,
    PredictionRequest,
    PredictionResponse,
)
from clarity.ml.models.manager import (
    LoadedModel,
    LoadingStrategy,
    ModelLoadConfig,
    ModelManager,
    ModelPerformanceMetrics,
    get_model_manager,
)
from clarity.ml.models.progressive_loader import (
    ApplicationPhase,
    ModelAvailabilityStatus,
    ProgressiveLoadingConfig,
    ProgressiveLoadingService,
    get_progressive_service,
    progressive_loading_lifespan,
)
from clarity.ml.models.registry import (
    LEGACY_PAT_MODELS,
    ModelAlias,
    ModelMetadata,
    ModelRegistry,
    ModelRegistryConfig,
    ModelStatus,
    ModelTier,
    initialize_legacy_models,
)

__all__ = [
    "LEGACY_PAT_MODELS",
    "ApplicationPhase",
    "LoadedModel",
    "LoadingStrategy",
    # Local Server
    "LocalModelServer",
    "MockPATModel",
    "ModelAlias",
    "ModelAvailabilityStatus",
    "ModelLoadConfig",
    # Manager
    "ModelManager",
    "ModelMetadata",
    "ModelPerformanceMetrics",
    # Registry
    "ModelRegistry",
    "ModelRegistryConfig",
    "ModelServerConfig",
    "ModelStatus",
    "ModelTier",
    "PredictionRequest",
    "PredictionResponse",
    "ProgressiveLoadingConfig",
    # Progressive Loader
    "ProgressiveLoadingService",
    "get_model_manager",
    "get_progressive_service",
    "initialize_legacy_models",
    "progressive_loading_lifespan",
]


# Version information
__version__ = "1.0.0"
__author__ = "Claude AI Assistant"
__description__ = "Revolutionary ML Model Management System for Clarity"
