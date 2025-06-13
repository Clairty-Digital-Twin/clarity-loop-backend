"""Tests f r integrations module initialization."""

from __future__ import annotations

import clarity.integrations
from clarity.integrations import AppleWatchDataProcessor, HealthKitClient


class TestIntegrationsInit:
    """Test the integrations module initialization."""

    def test_module_imports(self) -> None:
        """Test that integration classes can be imported from the module."""
        # Test that classes are available
        assert AppleWatchDataProcessor is not None
        assert HealthKitClient is not None

    def test_module_all_exports(self) -> None:
        """Test that __all__ exports are correctly defined."""
        # Check that __all__ contains expected exports
        assert hasattr(clarity.integrations, "__all__")
        assert "AppleWatchDataProcessor" in clarity.integrations.__all__
        assert "HealthKitClient" in clarity.integrations.__all__
        assert len(clarity.integrations.__all__) == 2

    def test_classes_are_importable(self) -> None:
        """Test that the exported classes can actually be instantiated or inspected."""
        # Test that these are actually classes/callables
        assert callable(AppleWatchDataProcessor)
        assert callable(HealthKitClient)

        # Test that they have expected characteristics of classes
        assert hasattr(AppleWatchDataProcessor, "__name__")
        assert hasattr(HealthKitClient, "__name__")
        assert AppleWatchDataProcessor.__name__ == "AppleWatchDataProcessor"
        assert HealthKitClient.__name__ == "HealthKitClient"
