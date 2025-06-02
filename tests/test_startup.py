"""CLARITY Digital Twin Platform - Startup Health Tests.

Tests to ensure the application starts quickly and properly without hanging.
These tests verify that the lifespan function works correctly with timeouts.
"""

import asyncio
from pathlib import Path
import sys
import time

import pytest

# Add src directory to Python path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestApplicationStartup:
    """Test suite for application startup performance and reliability."""

    @pytest.mark.asyncio
    @staticmethod
    async def test_application_starts_quickly() -> None:
        """Ensure app starts in under 10 seconds with proper timeout handling."""
        from clarity.core.container import create_application

        start_time = time.perf_counter()

        # Test that app creation doesn't hang
        app = await asyncio.wait_for(
            asyncio.to_thread(create_application),
            timeout=10.0,  # Should start much faster than this
        )

        elapsed = time.perf_counter() - start_time

        # Verify app was created successfully
        assert app is not None
        assert app.title == "CLARITY Digital Twin Platform"
        assert elapsed < 10.0, f"Startup took {elapsed:.2f}s - too slow!"

    @pytest.mark.asyncio
    @staticmethod
    async def test_lifespan_context_manager() -> None:
        """Test that the lifespan context manager works without hanging."""
        from fastapi import FastAPI

        from clarity.core.container import get_container

        container = get_container()

        # Create a minimal FastAPI app for testing
        test_app = FastAPI()

        start_time = time.perf_counter()

        # Test lifespan startup/shutdown cycle
        async with asyncio.timeout(15.0):  # Generous timeout for CI
            async with container.app_lifespan(test_app):
                elapsed_startup = time.perf_counter() - start_time

                # App should be ready for use
                assert (
                    elapsed_startup < 12.0
                ), f"Lifespan startup too slow: {elapsed_startup:.2f}s"

                # Brief operation to ensure everything works
                await asyncio.sleep(0.1)

        total_elapsed = time.perf_counter() - start_time
        assert (
            total_elapsed < 15.0
        ), f"Total lifespan cycle too slow: {total_elapsed:.2f}s"

    @staticmethod
    def test_config_provider_performance() -> None:
        """Test that config provider initialization is fast."""
        from clarity.core.container import get_container

        start_time = time.perf_counter()

        container = get_container()
        config = container.get_config_provider()

        # Test basic config access
        is_dev = config.is_development()
        should_skip = config.should_skip_external_services()
        auth_enabled = config.is_auth_enabled()

        elapsed = time.perf_counter() - start_time

        assert elapsed < 1.0, f"Config provider too slow: {elapsed:.2f}s"
        assert isinstance(is_dev, bool)
        assert isinstance(should_skip, bool)
        assert isinstance(auth_enabled, bool)

    @pytest.mark.asyncio
    @staticmethod
    async def test_dependency_injection_speed() -> None:
        """Test that dependency injection doesn't cause delays."""
        from clarity.core.container import get_container

        container = get_container()

        start_time = time.perf_counter()

        # Get all major dependencies
        config = container.get_config_provider()
        auth = container.get_auth_provider()
        repo = container.get_health_data_repository()

        elapsed = time.perf_counter() - start_time

        assert elapsed < 2.0, f"Dependency injection too slow: {elapsed:.2f}s"
        assert config is not None
        assert auth is not None
        assert repo is not None

    @pytest.mark.asyncio
    @staticmethod
    async def test_mock_services_in_development() -> None:
        """Test that mock services are used in development to prevent hangs."""
        from clarity.core.container import get_container

        container = get_container()
        config = container.get_config_provider()

        # In development or when skipping external services, should use mocks
        if config.should_skip_external_services():
            auth = container.get_auth_provider()
            repo = container.get_health_data_repository()

            # Should use mock implementations
            auth_type = type(auth).__name__
            repo_type = type(repo).__name__

            # Mock services should be available
            assert "Mock" in auth_type or "mock" in auth_type.lower()
            assert "Mock" in repo_type or "mock" in repo_type.lower()

    @pytest.mark.asyncio
    @staticmethod
    async def test_timeout_protection() -> None:
        """Test that startup has timeout protection."""
        from clarity.core.config import get_settings

        settings = get_settings()

        # Verify timeout configuration exists
        assert hasattr(settings, "startup_timeout")
        assert settings.startup_timeout > 0
        assert settings.startup_timeout <= 30  # Reasonable max

    @staticmethod
    def test_environment_validation() -> None:
        """Test that environment validation works correctly."""
        from clarity.core.config import get_settings

        settings = get_settings()

        # Should not raise exception in development
        assert settings.environment is not None
        assert isinstance(settings.skip_external_services, bool)

    @pytest.mark.asyncio
    @staticmethod
    async def test_graceful_failure_fallback() -> None:
        """Test that startup gracefully falls back to mock services on failure."""
        from clarity.core.container import get_container

        container = get_container()

        # Even if external services fail, we should get mock services
        start_time = time.perf_counter()

        try:
            auth = container.get_auth_provider()
            repo = container.get_health_data_repository()

            elapsed = time.perf_counter() - start_time

            # Should complete quickly regardless of external service availability
            assert elapsed < 5.0, f"Fallback too slow: {elapsed:.2f}s"
            assert auth is not None
            assert repo is not None

        except Exception as exc:
            pytest.fail(f"Startup should not fail completely: {exc}")


# Integration test for full startup cycle
class TestFullStartupCycle:
    """Integration tests for complete application startup."""

    @pytest.mark.asyncio
    @staticmethod
    async def test_complete_app_lifecycle() -> None:
        """Test complete application creation, startup, and shutdown."""
        from clarity.core.container import create_application

        # Full application lifecycle test
        start_time = time.perf_counter()

        # Create app (should be fast)
        app = create_application()
        creation_time = time.perf_counter() - start_time

        assert creation_time < 5.0, f"App creation too slow: {creation_time:.2f}s"

        # App should have basic properties
        assert app.title == "CLARITY Digital Twin Platform"
        assert hasattr(app, "router")


if __name__ == "__main__":
    # Allow running tests directly for debugging
    pytest.main([__file__, "-v", "--tb=short"])
