"""Integration module for enhanced feature flags with CLARITY application.

This module provides integration points for the enhanced feature flag system
with auto-refresh capabilities in the CLARITY Digital Twin Platform.
"""

import logging
import os
from typing import Any, Optional

from clarity.core.config_aws import Settings
from clarity.core.config_provider import ConfigProvider
from clarity.core.feature_flags_enhanced import (
    EnhancedFeatureFlagConfig,
    EnhancedFeatureFlagManager,
    RefreshMode,
    get_enhanced_feature_flag_manager,
)

logger = logging.getLogger(__name__)


def create_enhanced_feature_flag_manager(
    settings: Settings,
    config_provider: ConfigProvider | None = None,
) -> EnhancedFeatureFlagManager:
    """Create and configure enhanced feature flag manager for the application.

    Args:
        settings: Application settings
        config_provider: Optional configuration provider for remote config

    Returns:
        Configured enhanced feature flag manager
    """
    # Determine refresh mode based on environment
    if settings.environment == "production":
        refresh_mode = RefreshMode.BOTH  # Use both periodic and pub/sub in production
        refresh_interval = 60  # 1 minute
        stale_threshold = 300  # 5 minutes
    elif settings.environment == "staging":
        refresh_mode = RefreshMode.PERIODIC  # Periodic only in staging
        refresh_interval = 120  # 2 minutes
        stale_threshold = 600  # 10 minutes
    else:
        refresh_mode = RefreshMode.NONE  # No auto-refresh in development
        refresh_interval = 60
        stale_threshold = 3600  # 1 hour

    # Override from environment variables if present
    if os.getenv("FEATURE_FLAG_REFRESH_MODE"):
        refresh_mode = RefreshMode(os.getenv("FEATURE_FLAG_REFRESH_MODE"))

    refresh_interval = int(
        os.getenv("FEATURE_FLAG_REFRESH_INTERVAL", str(refresh_interval))
    )

    # Create enhanced configuration
    enhanced_config = EnhancedFeatureFlagConfig(
        refresh_interval_seconds=refresh_interval,
        refresh_mode=refresh_mode,
        redis_url=getattr(settings, "redis_url", "redis://localhost:6379"),
        pubsub_channel="clarity:feature_flags:config-changed",
        circuit_breaker_failure_threshold=3,
        circuit_breaker_recovery_timeout=30,
        stale_config_threshold_seconds=stale_threshold,
        enable_metrics=settings.environment in ("production", "staging"),
    )

    logger.info(
        "Creating enhanced feature flag manager: environment=%s, mode=%s, interval=%ds",
        settings.environment,
        refresh_mode.value,
        refresh_interval,
    )

    # Create manager
    manager = EnhancedFeatureFlagManager(
        config_provider=config_provider,
        enhanced_config=enhanced_config,
    )

    # Perform initial refresh if config provider is available
    if config_provider and refresh_mode != RefreshMode.NONE:
        try:
            success = manager.refresh()
            if success:
                logger.info("Initial feature flag refresh successful")
            else:
                logger.warning("Initial feature flag refresh failed, using defaults")
        except Exception as e:
            logger.error("Error during initial feature flag refresh: %s", e)

    return manager


def setup_feature_flags_for_app(app: Any, settings: Settings) -> EnhancedFeatureFlagManager:
    """Set up enhanced feature flags for FastAPI application.

    Args:
        app: FastAPI application instance
        settings: Application settings

    Returns:
        Enhanced feature flag manager instance
    """
    # Get or create config provider
    config_provider = None
    if hasattr(app.state, "config_provider"):
        config_provider = app.state.config_provider
    elif not settings.skip_external_services:
        # Create config provider if needed
        from clarity.core.config_provider import ConfigProvider

        config_provider = ConfigProvider(settings)

    # Create enhanced manager
    manager = create_enhanced_feature_flag_manager(settings, config_provider)

    # Store in app state for access
    app.state.feature_flag_manager = manager

    # Add startup/shutdown handlers
    @app.on_event("startup")  # type: ignore[misc]
    async def feature_flag_startup() -> None:
        """Initialize feature flag system on startup."""
        logger.info("Feature flag system started")

        # Log initial state
        if manager.is_config_stale():
            logger.warning("Feature flag configuration is stale at startup")

    @app.on_event("shutdown")  # type: ignore[misc]
    async def feature_flag_shutdown() -> None:
        """Cleanup feature flag system on shutdown."""
        logger.info("Shutting down feature flag system")
        manager.shutdown()

    return manager


# Convenience functions for common feature flag checks with auto-refresh


def is_mania_risk_enabled(user_id: str | None = None) -> bool:
    """Check if mania risk analysis is enabled.

    This function uses the enhanced manager with auto-refresh capabilities.

    Args:
        user_id: Optional user ID for personalized flags

    Returns:
        True if mania risk analysis is enabled
    """
    manager = get_enhanced_feature_flag_manager()

    # Check if config is stale and log warning
    if manager.is_config_stale():
        logger.warning(
            "Feature flag config is stale (age=%s seconds)",
            manager.get_config_age_seconds(),
        )

    return manager.is_enabled("mania_risk_analysis", user_id)


def is_pat_model_v2_enabled(user_id: str | None = None) -> bool:
    """Check if PAT model v2 is enabled.

    Args:
        user_id: Optional user ID for personalized flags

    Returns:
        True if PAT model v2 is enabled
    """
    manager = get_enhanced_feature_flag_manager()
    return manager.is_enabled("pat_model_v2", user_id)


def is_enhanced_security_enabled() -> bool:
    """Check if enhanced security features are enabled.

    Returns:
        True if enhanced security is enabled
    """
    manager = get_enhanced_feature_flag_manager()
    return manager.is_enabled("enhanced_security")


def get_feature_flag_health() -> dict[str, Any]:
    """Get health status of feature flag system.

    Returns:
        Dictionary with health information
    """
    manager = get_enhanced_feature_flag_manager()

    config_age = manager.get_config_age_seconds()

    return {
        "healthy": not manager.is_config_stale(),
        "config_age_seconds": config_age,
        "config_stale": manager.is_config_stale(),
        "circuit_breaker_state": manager.get_circuit_breaker_state(),
        "last_refresh": (
            manager._last_refresh_time.isoformat()
            if manager._last_refresh_time
            else None
        ),
        "refresh_failures": manager._refresh_failures,
        "refresh_mode": manager._enhanced_config.refresh_mode.value,
        "cache_size": len(manager._cache),
    }
