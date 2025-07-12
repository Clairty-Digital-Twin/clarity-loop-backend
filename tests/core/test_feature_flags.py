"""Tests for the feature flag system."""

import os
from unittest.mock import patch

import pytest

from clarity.core.feature_flags import (
    FeatureFlag,
    FeatureFlagManager,
    FeatureFlagStatus,
    feature_flag,
    get_feature_flag_manager,
    is_feature_enabled,
)


class TestFeatureFlagManager:
    """Test suite for FeatureFlagManager."""

    def test_initialization(self):
        """Test feature flag manager initialization."""
        manager = FeatureFlagManager()
        assert manager.config is not None
        assert "mania_risk_analysis" in manager.config.flags

    def test_mania_risk_flag_default_disabled(self):
        """Test that mania risk is disabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            manager = FeatureFlagManager()
            assert not manager.is_enabled("mania_risk_analysis")

    def test_mania_risk_flag_enabled_via_env(self):
        """Test that mania risk can be enabled via environment."""
        with patch.dict(os.environ, {"MANIA_RISK_ENABLED": "true"}):
            manager = FeatureFlagManager()
            assert manager.is_enabled("mania_risk_analysis")

    def test_unknown_flag_returns_default(self):
        """Test that unknown flags return default value."""
        manager = FeatureFlagManager()
        assert not manager.is_enabled("unknown_flag", default=False)
        assert manager.is_enabled("unknown_flag", default=True)

    def test_beta_user_access(self):
        """Test beta user access to features."""
        manager = FeatureFlagManager()

        # Add a beta feature
        beta_flag = FeatureFlag(
            name="beta_feature",
            status=FeatureFlagStatus.BETA,
            beta_users=["user123", "user456"],
        )
        manager.config.flags["beta_feature"] = beta_flag

        # Beta users should have access
        assert manager.is_enabled("beta_feature", user_id="user123")
        assert manager.is_enabled("beta_feature", user_id="user456")

        # Non-beta users should not have access
        assert not manager.is_enabled("beta_feature", user_id="user789")
        assert not manager.is_enabled("beta_feature")  # No user_id

    def test_canary_rollout(self):
        """Test canary rollout functionality."""
        manager = FeatureFlagManager()

        # Add a canary feature with 50% rollout
        canary_flag = FeatureFlag(
            name="canary_feature",
            status=FeatureFlagStatus.CANARY,
            rollout_percentage=50.0,
        )
        manager.config.flags["canary_feature"] = canary_flag

        # Test with multiple users - some should be enabled, some not
        enabled_count = 0
        for i in range(100):
            if manager.is_enabled("canary_feature", user_id=f"user{i}"):
                enabled_count += 1

        # Should be roughly 50% (allow for some variance)
        assert 30 <= enabled_count <= 70

    def test_caching(self):
        """Test that feature flag results are cached."""
        manager = FeatureFlagManager()

        # Clear cache first
        manager.clear_cache()

        # First call should not be cached
        result1 = manager.is_enabled("mania_risk_analysis", user_id="user123")

        # Modify the flag directly (bypassing normal flow)
        manager.config.flags["mania_risk_analysis"].status = FeatureFlagStatus.ENABLED

        # Second call should return cached result (not the modified value)
        result2 = manager.is_enabled("mania_risk_analysis", user_id="user123")
        assert result1 == result2

        # After clearing cache, should get new value
        manager.clear_cache()
        result3 = manager.is_enabled("mania_risk_analysis", user_id="user123")
        assert result3 is True

    def test_error_handling(self):
        """Test error handling in feature flag checks."""
        manager = FeatureFlagManager()

        # Create a mock that raises an exception when accessing flags
        with patch.object(manager.config, "flags") as mock_flags:
            mock_flags.get.side_effect = Exception("Test error")

            # Should return default value on error
            assert not manager.is_enabled("any_flag", default=False)
            assert manager.is_enabled("any_flag", default=True)


class TestFeatureFlagDecorator:
    """Test suite for feature flag decorator."""

    def test_decorator_enabled_feature(self):
        """Test decorator with enabled feature."""
        with patch.dict(os.environ, {"MANIA_RISK_ENABLED": "true"}):
            # Reset the global manager to pick up env change
            import clarity.core.feature_flags

            clarity.core.feature_flags._feature_flag_manager = None

            @feature_flag("mania_risk_analysis", fallback_value={"status": "disabled"})
            def process_mania_risk(user_id: str):
                return {"status": "enabled", "user": user_id}

            result = process_mania_risk(user_id="user123")
            assert result["status"] == "enabled"
            assert result["user"] == "user123"

    def test_decorator_disabled_feature(self):
        """Test decorator with disabled feature."""
        with patch.dict(os.environ, {}, clear=True):
            # Reset the global manager
            import clarity.core.feature_flags

            clarity.core.feature_flags._feature_flag_manager = None

            @feature_flag("mania_risk_analysis", fallback_value={"status": "disabled"})
            def process_mania_risk(user_id: str):
                return {"status": "enabled", "user": user_id}

            result = process_mania_risk(user_id="user123")
            assert result["status"] == "disabled"

    def test_decorator_no_fallback(self):
        """Test decorator with no fallback value."""
        with patch.dict(os.environ, {}, clear=True):
            # Reset the global manager
            import clarity.core.feature_flags

            clarity.core.feature_flags._feature_flag_manager = None

            @feature_flag("mania_risk_analysis")
            def process_mania_risk(user_id: str):
                return {"status": "enabled", "user": user_id}

            result = process_mania_risk(user_id="user123")
            assert result is None


class TestGlobalFunctions:
    """Test suite for global feature flag functions."""

    def test_get_feature_flag_manager_singleton(self):
        """Test that get_feature_flag_manager returns singleton."""
        manager1 = get_feature_flag_manager()
        manager2 = get_feature_flag_manager()
        assert manager1 is manager2

    def test_is_feature_enabled_convenience(self):
        """Test is_feature_enabled convenience function."""
        with patch.dict(os.environ, {"MANIA_RISK_ENABLED": "true"}):
            # Reset the global manager
            import clarity.core.feature_flags

            clarity.core.feature_flags._feature_flag_manager = None

            assert is_feature_enabled("mania_risk_analysis")
            assert not is_feature_enabled("unknown_feature", default=False)
