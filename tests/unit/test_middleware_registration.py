"""Test middleware registration in the dependency injection container.

Tests that the Firebase authentication middleware can be properly registered
without type errors, as specified in subtask 29.1.
"""

from fastapi.testclient import TestClient

from clarity.core.config import get_settings
from clarity.core.container import create_application


class TestMiddlewareRegistration:
    """Test suite for middleware registration functionality."""

    def test_container_initializes_without_type_errors(self) -> None:
        """Test that the application container initializes without type errors.

        This test verifies that the dependency injection container can be created
        and the FastAPI application can be instantiated without any type-related
        runtime errors, confirming that the middleware registration is properly
        typed.
        """
        # Create the application using the factory
        app = create_application()

        # Verify the app was created successfully
        assert app is not None
        assert app.title == "CLARITY Digital Twin Platform"

    def test_middleware_registration_with_auth_disabled(self) -> None:
        """Test middleware registration when authentication is disabled.

        This test ensures that when authentication is disabled in the configuration,
        the middleware registration code still executes without errors.
        """
        # Create test client to trigger app initialization
        app = create_application()
        client = TestClient(app)

        # Make a request to trigger middleware pipeline
        response = client.get("/health")

        # Should succeed without authentication middleware
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_middleware_registration_with_auth_enabled(self) -> None:
        """Test middleware registration when authentication is enabled.

        This test verifies that the Firebase authentication middleware can be
        registered when authentication is enabled, even if it falls back to
        mock authentication due to missing credentials.
        """
        # Override settings to enable auth for this test
        settings = get_settings()
        original_auth_enabled = getattr(settings, "enable_auth", False)

        try:
            # Temporarily enable auth
            settings.enable_auth = True

            # Create application with auth enabled
            app = create_application()
            client = TestClient(app)

            # Make a request to health endpoint (should still work - exempt path)
            response = client.get("/health")
            assert response.status_code == 200

        finally:
            # Restore original setting
            settings.enable_auth = original_auth_enabled

    def test_app_can_be_created_multiple_times(self) -> None:
        """Test that the application factory can be called multiple times.

        This verifies that the middleware registration doesn't cause issues
        with multiple application instances.
        """
        app1 = create_application()
        app2 = create_application()

        assert app1 is not None
        assert app2 is not None

        # They should be different instances
        assert app1 is not app2
