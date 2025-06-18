"""Integration tests for account lockout with authentication endpoints."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import pytest

from clarity.auth.lockout_service import AccountLockoutError
from clarity.main import app


class TestAuthLockoutIntegration:
    """Test lockout service integration with auth endpoints."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        # Force mock services for integration tests
        import os
        original_skip = os.environ.get("SKIP_EXTERNAL_SERVICES")
        os.environ["SKIP_EXTERNAL_SERVICES"] = "true"
        
        yield TestClient(app)
        
        # Restore original value
        if original_skip is None:
            os.environ.pop("SKIP_EXTERNAL_SERVICES", None)
        else:
            os.environ["SKIP_EXTERNAL_SERVICES"] = original_skip

    @pytest.mark.asyncio
    async def test_lockout_triggers_after_failed_attempts(
        self, client: TestClient
    ) -> None:
        """Test that lockout service is called during failed login attempts."""
        # Mock the auth provider to always return invalid credentials
        with patch("clarity.api.v1.auth.get_auth_provider") as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.authenticate.side_effect = Exception("Invalid credentials")
            mock_get_provider.return_value = mock_provider

            # Mock the lockout service to track calls
            with patch("clarity.api.v1.auth.get_lockout_service") as mock_get_lockout:
                mock_lockout = AsyncMock()
                mock_lockout.check_lockout = AsyncMock()
                mock_lockout.record_failed_attempt = AsyncMock()
                mock_get_lockout.return_value = mock_lockout

                # Attempt login with bad credentials
                response = client.post(
                    "/api/v1/auth/login",
                    json={"email": "test@example.com", "password": "wrongpassword"},
                )

                # Should return 401 for invalid credentials
                assert response.status_code == 401

                # Verify lockout service was called
                mock_lockout.check_lockout.assert_called_once_with("test@example.com")
                mock_lockout.record_failed_attempt.assert_called_once()

    @pytest.mark.asyncio
    async def test_lockout_blocks_login_attempt(self, client: TestClient) -> None:
        """Test that lockout exception blocks login attempts."""
        # Mock the lockout service to raise lockout error
        with patch("clarity.api.v1.auth.get_lockout_service") as mock_get_lockout:
            mock_lockout = AsyncMock()
            mock_lockout.check_lockout.side_effect = AccountLockoutError(
                "test@example.com", datetime.now() + timedelta(minutes=15)
            )
            mock_get_lockout.return_value = mock_lockout

            # Attempt login
            response = client.post(
                "/api/v1/auth/login",
                json={"email": "test@example.com", "password": "anypassword"},
            )

            # Should return 429 for account locked
            assert response.status_code == 429
            assert "account is locked" in response.json()["detail"]["detail"].lower()

    @pytest.mark.asyncio
    async def test_successful_login_resets_attempts(self, client: TestClient) -> None:
        """Test that successful login resets failed attempts."""
        # Enable self-signup and skip external services for testing
        import os
        os.environ["ENABLE_SELF_SIGNUP"] = "true"
        os.environ["SKIP_EXTERNAL_SERVICES"] = "true"
        
        with patch("clarity.api.v1.auth.get_auth_provider") as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.authenticate.return_value = {
                "access_token": "fake_token",
                "refresh_token": "fake_refresh_token",
                "token_type": "bearer",
                "expires_in": 3600,
                "user_id": "user123",
                "email": "test@example.com",
            }
            mock_get_provider.return_value = mock_provider

            # Mock the lockout service dependency
            with patch("clarity.api.v1.auth.get_lockout_service") as mock_get_lockout:
                mock_lockout = AsyncMock()
                mock_lockout.check_lockout = AsyncMock()
                mock_lockout.reset_attempts = AsyncMock()
                mock_get_lockout.return_value = mock_lockout

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
