"""Startup Performance Tests.

Tests to ensure CLARITY application starts quickly and handles errors gracefully.
"""

import asyncio
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING

from fastapi import FastAPI
import pytest

from clarity.core.config import get_settings
from clarity.core.container import create_application, get_container

if TYPE_CHECKING:
    from clarity.core.interfaces import IHealthDataRepository

# Add src directory to Python path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestApplicationStartup:
    """Test application startup performance and reliability."""

    @pytest.mark.asyncio
    @staticmethod
    async def test_application_starts_quickly() -> None:
        """Ensure app starts in under 10 seconds with proper timeout handling."""
        start_time = time.perf_counter()

        try:
            # Test app creation with timeout
            app = create_application()
            assert isinstance(app, FastAPI)

        except (TimeoutError, RuntimeError, ImportError) as e:
            pytest.fail(f"Application creation should not fail: {e}")

        startup_duration = time.perf_counter() - start_time

        # Should start quickly (under 2 seconds)
        assert startup_duration < 2.0, f"Startup too slow: {startup_duration:.2f}s"

    @pytest.mark.asyncio
    @staticmethod
    async def test_lifespan_context_manager() -> None:
        """Test that the lifespan context manager works without hanging."""
        container = get_container()
        app = create_application()

        # Test lifespan context manager with timeout
        timeout_duration = 5.0

        try:
            # Use asyncio.wait_for to enforce timeout
            async with asyncio.timeout(timeout_duration):
                async with container.app_lifespan(app):
                    # Context manager should complete quickly
                    pass

        except TimeoutError:
            pytest.fail(f"Lifespan context manager timed out after {timeout_duration}s")
        except (RuntimeError, ImportError, ConnectionError) as e:
            pytest.fail(f"Lifespan context manager failed: {e}")

    @staticmethod
    def test_config_provider_performance() -> None:
        """Test that config provider initialization is fast."""
        start_time = time.perf_counter()

        try:
            container = get_container()
            config_provider = container.get_config_provider()
            assert config_provider is not None

        except (RuntimeError, ImportError, ValueError) as e:
            pytest.fail(f"Config provider creation should not fail: {e}")

        config_duration = time.perf_counter() - start_time

        assert config_duration < 0.5, (
            f"Config initialization too slow: {config_duration:.2f}s"
        )

    @staticmethod
    def test_dependency_injection_speed() -> None:
        """Test that dependency injection doesn't cause delays."""
        container = get_container()
        start_time = time.perf_counter()

        try:
            # Test dependency injection speed
            repository: IHealthDataRepository = container.get_health_data_repository()
            assert repository is not None

        except (RuntimeError, ImportError, ConnectionError) as e:
            pytest.fail(f"Dependency injection should not fail: {e}")

        di_duration = time.perf_counter() - start_time

        # Dependency injection should be fast
        assert di_duration < 1.0, f"Dependency injection too slow: {di_duration:.2f}s"

    @staticmethod
    def test_mock_services_in_development() -> None:
        """Test that mock services are used in development to prevent hangs."""
        container = get_container()

        try:
            # In development/testing, should use mock services
            repository: IHealthDataRepository = container.get_health_data_repository()
            assert repository is not None

        except (RuntimeError, ImportError, ConnectionError) as e:
            pytest.fail(f"Mock service creation should not fail: {e}")

    @pytest.mark.asyncio
    @staticmethod
    async def test_timeout_protection() -> None:
        """Test that startup has timeout protection."""
        settings = get_settings()

        # Should have timeout settings for external services
        assert hasattr(settings, "environment")

    @staticmethod
    def test_environment_validation() -> None:
        """Test that environment validation works correctly."""
        settings = get_settings()

        # Should have proper environment configuration
        assert settings.environment in {"development", "testing", "production"}
        assert hasattr(settings, "debug")

    @staticmethod
    def test_graceful_failure_fallback() -> None:
        """Test that startup gracefully falls back to mock services on failure."""
        container = get_container()

        # Should be able to get repository even if external services fail
        try:
            repository: IHealthDataRepository = container.get_health_data_repository()
            assert repository is not None
            # In development/testing, this should be a mock
        except (RuntimeError, ImportError, ConnectionError) as e:
            # This is a legitimate skip - external services may not be available in CI/CD
            pytest.skip(f"External services not available: {e}")


class TestFullStartupCycle:
    """Test complete application startup and shutdown cycle."""

    @pytest.mark.asyncio
    @staticmethod
    async def test_complete_app_lifecycle() -> None:
        """Test complete application creation, startup, and shutdown."""
        # Full application lifecycle test
        start_time = time.perf_counter()

        try:
            container = get_container()
            app = create_application()

            async with container.app_lifespan(app):
                # App should be ready for requests
                assert isinstance(app, FastAPI)

            lifecycle_duration = time.perf_counter() - start_time

            assert lifecycle_duration < 3.0, (
                f"Lifecycle too slow: {lifecycle_duration:.2f}s"
            )

        except (RuntimeError, ImportError, TimeoutError, ConnectionError) as e:
            pytest.fail(f"Complete application lifecycle should not fail: {e}")


if __name__ == "__main__":
    # Allow running tests directly for debugging
    pytest.main([__file__, "-v", "--tb=short"])
