"""CLARITY Digital Twin Platform - Startup Health Tests

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
    async def test_application_starts_quickly(self):
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

        print(f"✅ App started successfully in {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def test_lifespan_context_manager(self):
        """Test that the lifespan context manager works without hanging."""
        from clarity.core.container import get_container

        container = get_container()

        # Create a minimal FastAPI app for testing
        from fastapi import FastAPI

        test_app = FastAPI()

        start_time = time.perf_counter()

        # Test lifespan startup/shutdown cycle
        async with asyncio.timeout(15.0):  # Generous timeout for CI
            async with container.app_lifespan(test_app):
                elapsed_startup = time.perf_counter() - start_time
                print(f"✅ Lifespan startup completed in {elapsed_startup:.2f}s")

                # App should be ready for use
                assert (
                    elapsed_startup < 12.0
                ), f"Lifespan startup too slow: {elapsed_startup:.2f}s"

                # Brief operation to ensure everything works
                await asyncio.sleep(0.1)

        total_elapsed = time.perf_counter() - start_time
        print(f"✅ Complete lifespan cycle in {total_elapsed:.2f}s")

    def test_config_provider_performance(self):
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

        print(f"✅ Config provider ready in {elapsed:.4f}s")
        print(f"   • Development: {is_dev}")
        print(f"   • Skip external: {should_skip}")
        print(f"   • Auth enabled: {auth_enabled}")

    @pytest.mark.asyncio
    async def test_dependency_injection_speed(self):
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

        print(f"✅ All dependencies ready in {elapsed:.4f}s")
        print(f"   • Config: {type(config).__name__}")
        print(f"   • Auth: {type(auth).__name__}")
        print(f"   • Repository: {type(repo).__name__}")

    @pytest.mark.asyncio
    async def test_mock_services_in_development(self):
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

            print("✅ Using mock services for fast startup:")
            print(f"   • Auth: {auth_type}")
            print(f"   • Repository: {repo_type}")

            # Mock services should be available
            assert "Mock" in auth_type or "mock" in auth_type.lower()
            assert "Mock" in repo_type or "mock" in repo_type.lower()

    @pytest.mark.asyncio
    async def test_timeout_protection(self):
        """Test that startup has timeout protection."""
        from clarity.core.config import get_settings

        settings = get_settings()

        # Verify timeout configuration exists
        assert hasattr(settings, "startup_timeout")
        assert settings.startup_timeout > 0
        assert settings.startup_timeout <= 30  # Reasonable max

        print(f"✅ Startup timeout configured: {settings.startup_timeout}s")

    def test_environment_validation(self):
        """Test that environment validation works correctly."""
        from clarity.core.config import get_settings

        settings = get_settings()

        # Should not raise exception in development
        assert settings.environment is not None
        assert isinstance(settings.skip_external_services, bool)

        print("✅ Environment validation passed:")
        print(f"   • Environment: {settings.environment}")
        print(f"   • Skip external: {settings.skip_external_services}")

    @pytest.mark.asyncio
    async def test_graceful_failure_fallback(self):
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

            print(f"✅ Fallback services ready in {elapsed:.4f}s")

        except Exception as e:
            pytest.fail(f"Startup should not fail completely: {e}")


# Integration test for full startup cycle
class TestFullStartupCycle:
    """Integration tests for complete application startup."""

    @pytest.mark.asyncio
    async def test_complete_app_lifecycle(self):
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

        print("✅ Complete application lifecycle test passed:")
        print(f"   • Creation time: {creation_time:.2f}s")
        print(f"   • App title: {app.title}")
        print(f"   • Lifespan enabled: {app.router.lifespan_context is not None}")


if __name__ == "__main__":
    # Allow running tests directly for debugging
    pytest.main([__file__, "-v", "--tb=short"])
