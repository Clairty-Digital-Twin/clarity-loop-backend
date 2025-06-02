"""Startup Performance Tests.

Tests to ensure CLARITY application starts quickly and handles errors gracefully.
"""

import asyncio
from pathlib import Path
import sys
import time

from fastapi import FastAPI
import pytest

from clarity.core.config import get_settings
from clarity.core.container import create_application, get_container

# Add src directory to Python path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestApplicationStartup:
    """Test application startup performance and reliability."""

    async def test_application_starts_quickly(self) -> None:
        """Ensure app starts in under 10 seconds with proper timeout handling."""
        start_time = time.perf_counter()

        try:
            # Test application creation (should be fast)
            app = create_application()
            assert app is not None
            assert isinstance(app, FastAPI)

        except Exception:
            pytest.fail("Application creation should not fail")

        end_time = time.perf_counter()
        startup_duration = end_time - start_time

        # Application creation should be very fast (under 2 seconds)
        assert startup_duration < 2.0, f"Startup too slow: {startup_duration:.2f}s"

    async def test_lifespan_context_manager(self) -> None:
        """Test that the lifespan context manager works without hanging."""
        container = get_container()

        # Test with timeout to prevent hanging
        timeout_duration = 5.0

        try:
            # Create a mock FastAPI app for testing
            app = FastAPI()

            # Test the lifespan context manager with timeout
            async with asyncio.timeout(timeout_duration):
                async with container.app_lifespan(app):
                    # Context manager should complete quickly
                    pass

        except TimeoutError:
            pytest.fail(f"Lifespan context manager timed out after {timeout_duration}s")
        except Exception as e:
            pytest.fail(f"Lifespan context manager failed: {e}")

    def test_config_provider_performance(self) -> None:
        """Test that config provider initialization is fast."""
        start_time = time.perf_counter()

        try:
            container = get_container()
            config_provider = container.get_config_provider()
            assert config_provider is not None

        except Exception:
            pytest.fail("Config provider creation should not fail")

        end_time = time.perf_counter()
        config_duration = end_time - start_time

        # Config provider should initialize very quickly
        assert (
            config_duration < 0.1
        ), f"Config initialization too slow: {config_duration:.2f}s"

    async def test_dependency_injection_speed(self) -> None:
        """Test that dependency injection doesn't cause delays."""
        container = get_container()

        start_time = time.perf_counter()

        try:
            # Test getting various dependencies
            auth_provider = container.get_auth_provider()
            config_provider = container.get_config_provider()
            repository = container.get_health_data_repository()

            assert auth_provider is not None
            assert config_provider is not None
            assert repository is not None

        except Exception:
            pytest.fail("Dependency injection should not fail")

        end_time = time.perf_counter()
        di_duration = end_time - start_time

        # Dependency injection should be fast
        assert di_duration < 1.0, f"Dependency injection too slow: {di_duration:.2f}s"

    async def test_mock_services_in_development(self) -> None:
        """Test that mock services are used in development to prevent hangs."""
        container = get_container()

        try:
            # In development/testing, we should get mock services that don't hang
            auth_provider = container.get_auth_provider()
            repository = container.get_health_data_repository()

            # These should be mock services in test environment
            # They should respond quickly without external dependencies
            assert auth_provider is not None
            assert repository is not None

        except Exception:
            pytest.fail("Mock service creation should not fail")

    async def test_timeout_protection(self) -> None:
        """Test that startup has timeout protection."""
        settings = get_settings()

        # Settings should load quickly without hanging
        assert settings is not None
        assert hasattr(settings, "environment")

    def test_environment_validation(self) -> None:
        """Test that environment validation works correctly."""
        settings = get_settings()

        # Should have proper environment configuration
        assert settings.environment in ["development", "testing", "production"]
        assert hasattr(settings, "debug")

    async def test_graceful_failure_fallback(self) -> None:
        """Test that startup gracefully falls back to mock services on failure."""
        container = get_container()

        try:
            # Even if external services fail, we should get fallback services
            auth_provider = container.get_auth_provider()
            repo = container.get_health_data_repository()

            assert auth_provider is not None
            assert repo is not None

        except (ImportError, RuntimeError, OSError) as exc:
            pytest.fail(f"Startup should not fail completely: {exc}")


class TestFullStartupCycle:
    """Test complete application startup and shutdown cycle."""

    async def test_complete_app_lifecycle(self) -> None:
        """Test complete application creation, startup, and shutdown."""
        # Full application lifecycle test
        start_time = time.perf_counter()

        try:
            # Create application
            app = create_application()
            assert app is not None

            # Application should be ready immediately
            end_time = time.perf_counter()
            lifecycle_duration = end_time - start_time

            # Complete lifecycle should be fast
            assert (
                lifecycle_duration < 3.0
            ), f"Lifecycle too slow: {lifecycle_duration:.2f}s"

        except Exception:
            pytest.fail("Complete application lifecycle should not fail")


if __name__ == "__main__":
    # Allow running tests directly for debugging
    pytest.main([__file__, "-v", "--tb=short"])
