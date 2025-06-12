"""Basic tests for Health Data API endpoints.

Tests cover:
- Basic API endpoint structure validation
- Simple functionality tests
"""

from __future__ import annotations

from clarity.api.v1.health_data import router


class TestHealthDataAPIBasics:
    """Test basic API functionality."""

    @staticmethod
    def test_api_module_imports() -> None:
        """Test that API module imports correctly."""
        assert router is not None

    @staticmethod
    def test_router_exists() -> None:
        """Test that router is configured."""
        # Check router has routes
        assert len(router.routes) > 0

    @staticmethod
    def test_basic_endpoint_structure() -> None:
        """Test that endpoints are configured."""
        routes = [getattr(route, "path", str(route)) for route in router.routes]  # type: ignore[attr-defined]

        # Should have health check endpoint at minimum
        assert any("/health" in path for path in routes)
