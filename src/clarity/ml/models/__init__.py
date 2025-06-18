"""
Revolutionary ML Model Management System

This package provides a comprehensive ML model management solution with:
- Intelligent model registry with versioning
- Progressive loading with caching
- Local development server
- Performance monitoring
- CLI management tools
"""

from .registry import (
    ModelRegistry,
    ModelRegistryConfig,
    ModelMetadata,
    ModelStatus,
    ModelTier,
    ModelAlias,
    LEGACY_PAT_MODELS,
    initialize_legacy_models
)

from .manager import (
    ModelManager,
    ModelLoadConfig,
    LoadingStrategy,
    LoadedModel,
    ModelPerformanceMetrics,
    get_model_manager
)

from .progressive_loader import (
    ProgressiveLoadingService,
    ProgressiveLoadingConfig,
    ApplicationPhase,
    ModelAvailabilityStatus,
    get_progressive_service,
    progressive_loading_lifespan
)

from .local_server import (
    LocalModelServer,
    ModelServerConfig,
    PredictionRequest,
    PredictionResponse,
    MockPATModel
)


__all__ = [
    # Registry
    "ModelRegistry",
    "ModelRegistryConfig", 
    "ModelMetadata",
    "ModelStatus",
    "ModelTier",
    "ModelAlias",
    "LEGACY_PAT_MODELS",
    "initialize_legacy_models",
    
    # Manager
    "ModelManager",
    "ModelLoadConfig",
    "LoadingStrategy",
    "LoadedModel",
    "ModelPerformanceMetrics",
    "get_model_manager",
    
    # Progressive Loader
    "ProgressiveLoadingService",
    "ProgressiveLoadingConfig",
    "ApplicationPhase",
    "ModelAvailabilityStatus",
    "get_progressive_service",
    "progressive_loading_lifespan",
    
    # Local Server
    "LocalModelServer",
    "ModelServerConfig",
    "PredictionRequest", 
    "PredictionResponse",
    "MockPATModel"
]


# Version information
__version__ = "1.0.0"
__author__ = "Claude AI Assistant"
__description__ = "Revolutionary ML Model Management System for Clarity"