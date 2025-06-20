"""Integration tests for account lockout with authentication endpoints."""

from datetime import UTC, datetime, timedelta
import os
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from clarity.api.v1.auth import get_auth_provider, get_lockout_service
from clarity.auth.aws_cognito_provider import CognitoAuthProvider
from clarity.auth.lockout_service import AccountLockoutError
from clarity.core.exceptions import InvalidCredentialsError
from clarity.main import create_app


class TestAuthLockoutIntegration:
    """Test lockout service integration with auth endpoints."""

    @pytest.fixture
    def app_and_client(self) -> tuple[FastAPI, TestClient]:
        """Create test app and client."""
        # Force mock services for integration tests
        original_skip = os.environ.get("SKIP_EXTERNAL_SERVICES")
        os.environ["SKIP_EXTERNAL_SERVICES"] = "true"

        app = create_app()
        yield app, TestClient(app)

        # Restore original value
        if original_skip is None:
            os.environ.pop("SKIP_EXTERNAL_SERVICES", None)
        else:
            os.environ["SKIP_EXTERNAL_SERVICES"] = original_skip

    @pytest.mark.asyncio
    async def test_lockout_triggers_after_failed_attempts(
        self, app_and_client: tuple[FastAPI, TestClient]
    ) -> None:
        """Test that lockout service is called during failed login attempts."""
        app, client = app_and_client
        # Override the app's dependency injection
        # Create mock that is instance of CognitoAuthProvider
        mock_provider = AsyncMock(spec=CognitoAuthProvider)
        mock_provider.authenticate.side_effect = InvalidCredentialsError(
            "Invalid credentials"
        )

        mock_lockout = AsyncMock()
        mock_lockout.check_lockout = AsyncMock()
        mock_lockout.record_failed_attempt = AsyncMock()

        # Override dependencies
        app.dependency_overrides[get_auth_provider] = lambda: mock_provider
        app.dependency_overrides[get_lockout_service] = lambda: mock_lockout

        try:
            # Attempt login with bad credentials
            response = client.post(
                "/api/v1/auth/login",
                json={"email": "test@example.com", "password": "wrongpassword"},
            )

            # Check response if not 401
            if response.status_code != 401:
                pass

            # Should return 401 for invalid credentials
            assert response.status_code == 401

            # Verify lockout service was called
            mock_lockout.check_lockout.assert_called_once_with("test@example.com")
            mock_lockout.record_failed_attempt.assert_called_once()
        finally:
            # Clear overrides
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_lockout_blocks_login_attempt(
        self, app_and_client: tuple[FastAPI, TestClient]
    ) -> None:
        """Test that lockout exception blocks login attempts."""
        app, client = app_and_client
        # Create mock lockout service
        mock_lockout = AsyncMock()
        mock_lockout.check_lockout.side_effect = AccountLockoutError(
            "test@example.com", datetime.now(UTC) + timedelta(minutes=15)
        )

        # Override dependency
        app.dependency_overrides[get_lockout_service] = lambda: mock_lockout

        try:
            # Attempt login
            response = client.post(
                "/api/v1/auth/login",
                json={"email": "test@example.com", "password": "anypassword"},
            )

            # Should return 429 for account locked
            assert response.status_code == 429
            assert "is locked" in response.json()["detail"]["detail"].lower()
        finally:
            # Clear overrides
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_successful_login_resets_attempts(
        self, app_and_client: tuple[FastAPI, TestClient]
    ) -> None:
        """Test that successful login resets failed attempts."""
        app, client = app_and_client
        # Enable self-signup and skip external services for testing
        os.environ["ENABLE_SELF_SIGNUP"] = "true"
        os.environ["SKIP_EXTERNAL_SERVICES"] = "true"

        # Create mock that is instance of CognitoAuthProvider
        mock_provider = AsyncMock(spec=CognitoAuthProvider)
        mock_provider.authenticate.return_value = {
            "access_token": "fake_token",
            "refresh_token": "fake_refresh_token",
            "token_type": "bearer",
            "expires_in": 3600,
            "user_id": "user123",
            "email": "test@example.com",
        }

        mock_lockout = AsyncMock()
        mock_lockout.check_lockout = AsyncMock()
        mock_lockout.reset_attempts = AsyncMock()

        # Override dependencies
        app.dependency_overrides[get_auth_provider] = lambda: mock_provider
        app.dependency_overrides[get_lockout_service] = lambda: mock_lockout

        try:
            # Successful login
            response = client.post(
                "/api/v1/auth/login",
                json={"email": "test@example.com", "password": "correctpassword"},
            )

            # Should return 200 for successful login
            assert response.status_code == 200
            assert "access_token" in response.json()

            # Verify lockout service was called
            mock_lockout.check_lockout.assert_called_once_with("test@example.com")
            mock_lockout.reset_attempts.assert_called_once_with("test@example.com")
        finally:
            # Clear overrides
            app.dependency_overrides.clear()
