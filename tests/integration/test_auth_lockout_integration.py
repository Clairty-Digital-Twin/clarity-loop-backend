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
        return TestClient(app)

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
            with patch("clarity.api.v1.auth.lockout_service") as mock_lockout:
                mock_lockout.check_lockout = AsyncMock()
                mock_lockout.record_failed_attempt = AsyncMock()

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
        with patch("clarity.api.v1.auth.lockout_service") as mock_lockout:
            mock_lockout.check_lockout.side_effect = AccountLockoutError(
                "test@example.com", datetime.now() + timedelta(minutes=15)
            )

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
        # Mock successful authentication
        with patch("clarity.api.v1.auth.get_auth_provider") as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.authenticate.return_value = {
                "access_token": "fake_token",
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
