"""Feature Flag System for CLARITY Digital Twin Platform.

This module provides a centralized, production-ready feature flag system
with proper environment-based configuration and graceful degradation.
"""

from collections.abc import Callable
from enum import StrEnum
import logging
import os
from typing import Any, TypeVar, cast

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FeatureFlagStatus(StrEnum):
    """Feature flag status enum."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    BETA = "beta"  # Enabled for beta users only
    CANARY = "canary"  # Gradual rollout


class FeatureFlag(BaseModel):
    """Individual feature flag configuration."""

    name: str = Field(description="Unique feature flag name")
    status: FeatureFlagStatus = Field(default=FeatureFlagStatus.DISABLED)
    description: str = Field(default="")
    rollout_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    beta_users: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeatureFlagConfig(BaseModel):
    """Feature flag configuration container."""

    flags: dict[str, FeatureFlag] = Field(default_factory=dict)
    environment: str = Field(default="development")


class FeatureFlagManager:
    """Centralized feature flag manager with environment-aware configuration."""

    def __init__(self) -> None:
        self.config = self._load_configuration()
        self._cache: dict[str, bool] = {}

    def _load_configuration(self) -> FeatureFlagConfig:
        """Load feature flag configuration from environment and defaults."""
        environment = os.getenv("ENVIRONMENT", "development")

        # Define default feature flags
        default_flags = {
            "mania_risk_analysis": FeatureFlag(
                name="mania_risk_analysis",
                status=self._get_mania_risk_status(),
                description="Enable mania risk analysis module",
                rollout_percentage=100.0 if environment == "production" else 0.0,
            ),
            "pat_model_v2": FeatureFlag(
                name="pat_model_v2",
                status=FeatureFlagStatus.ENABLED,
                description="Use PAT model v2 for actigraphy analysis",
            ),
            "enhanced_security": FeatureFlag(
                name="enhanced_security",
                status=(
                    FeatureFlagStatus.ENABLED
                    if environment == "production"
                    else FeatureFlagStatus.DISABLED
                ),
                description="Enable enhanced security checks",
            ),
            "graceful_degradation": FeatureFlag(
                name="graceful_degradation",
                status=FeatureFlagStatus.ENABLED,
                description="Enable graceful degradation for failed services",
            ),
        }

        return FeatureFlagConfig(
            flags=default_flags,
            environment=environment,
        )

    def _get_mania_risk_status(self) -> FeatureFlagStatus:
        """Get mania risk feature status from environment."""
        # Check environment variable
        mania_enabled = os.getenv("MANIA_RISK_ENABLED", "false").lower() == "true"

        if mania_enabled:
            return FeatureFlagStatus.ENABLED
        return FeatureFlagStatus.DISABLED

    def is_enabled(
        self, flag_name: str, user_id: str | None = None, default: bool = False
    ) -> bool:
        """Check if a feature flag is enabled.

        Args:
            flag_name: Name of the feature flag
            user_id: Optional user ID for beta/canary checks
            default: Default value if flag not found

        Returns:
            True if feature is enabled, False otherwise
        """
        # Check cache first
        cache_key = f"{flag_name}:{user_id or 'global'}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            flag = self.config.flags.get(flag_name)
            if not flag:
                logger.warning(
                    f"Unknown feature flag: {flag_name}, using default: {default}"
                )
                return default

            # Check flag status
            if flag.status == FeatureFlagStatus.DISABLED:
                result = False
            elif flag.status == FeatureFlagStatus.ENABLED:
                result = True
            elif flag.status == FeatureFlagStatus.BETA and user_id:
                result = user_id in flag.beta_users
            elif flag.status == FeatureFlagStatus.CANARY:
                # Simple hash-based canary rollout
                if user_id:
                    user_hash = hash(user_id) % 100
                    result = user_hash < flag.rollout_percentage
                else:
                    result = False
            else:
                result = default

            # Cache the result
            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.exception(f"Error checking feature flag {flag_name}: {e}")
            return default

    def get_flag(self, flag_name: str) -> FeatureFlag | None:
        """Get feature flag configuration."""
        return self.config.flags.get(flag_name)

    def clear_cache(self) -> None:
        """Clear the feature flag cache."""
        self._cache.clear()


# Global instance
_feature_flag_manager: FeatureFlagManager | None = None


def get_feature_flag_manager() -> FeatureFlagManager:
    """Get or create the global feature flag manager."""
    global _feature_flag_manager
    if _feature_flag_manager is None:
        _feature_flag_manager = FeatureFlagManager()
    return _feature_flag_manager


def is_feature_enabled(
    flag_name: str, user_id: str | None = None, default: bool = False
) -> bool:
    """Convenience function to check if a feature is enabled.

    Args:
        flag_name: Name of the feature flag
        user_id: Optional user ID for beta/canary checks
        default: Default value if flag not found

    Returns:
        True if feature is enabled, False otherwise
    """
    manager = get_feature_flag_manager()
    return manager.is_enabled(flag_name, user_id, default)


def feature_flag(
    flag_name: str, default: bool = False, fallback_value: Any = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for feature flag controlled functions.

    Args:
        flag_name: Name of the feature flag
        default: Default value if flag not found
        fallback_value: Value to return if feature is disabled

    Usage:
        @feature_flag("new_algorithm", fallback_value={"result": "default"})
        def process_data(data):
            # New algorithm implementation
            return {"result": "processed"}
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Extract user_id if available in kwargs
            user_id = kwargs.get("user_id")

            if is_feature_enabled(flag_name, user_id, default):
                return func(*args, **kwargs)
            logger.debug(f"Feature {flag_name} is disabled, returning fallback")
            if fallback_value is not None:
                return cast(T, fallback_value)
            # Return None if no fallback specified
            return cast(T, None)

        return wrapper

    return decorator
