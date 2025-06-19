"""Tests for authentication API endpoints that actually test real code."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# Import the REAL modules we want to test
from clarity.api.v1.auth import router
from clarity.auth.aws_cognito_provider import CognitoAuthProvider
from clarity.auth.dependencies import get_auth_provider, get_lockout_service
from clarity.core.constants import (
    AUTH_HEADER_TYPE_BEARER,
    AUTH_SCOPE_FULL_ACCESS,
)
from clarity.core.exceptions import (
    EmailNotVerifiedError,
    InvalidCredentialsError,
    UserAlreadyExistsError,
)
from clarity.models.user import User


@pytest.fixture
def mock_cognito_provider() -> Mock:
    """Create a properly mocked Cognito provider."""
    provider = Mock(spec=CognitoAuthProvider)
    provider.authenticate = AsyncMock()
    provider.create_user = AsyncMock()
    provider.update_user = AsyncMock()
    provider.verify_token = AsyncMock()
    provider.get_user = AsyncMock()
    provider.client_id = "test1234567890"

    # Mock the cognito client
    mock_client = MagicMock()
    mock_client.exceptions.NotAuthorizedException = Exception
    provider.cognito_client = mock_client

    return provider


@pytest.fixture
def test_user() -> User:
    """Create test user."""
    return User(
        uid=str(uuid.uuid4()),
        email="test@example.com",
        display_name="Test User",
    )


@pytest.fixture
def auth_tokens() -> dict[str, Any]:
    """Create test auth tokens."""
    return {
        "access_token": "test-access-token",
        "refresh_token": "test-refresh-token",
        "expires_in": 3600,
    }


@pytest.fixture
def app(mock_cognito_provider: Mock) -> FastAPI:
    """Create a real FastAPI app that uses the actual auth router."""
    # Enable self-signup for tests
    os.environ["ENABLE_SELF_SIGNUP"] = "true"

    app = FastAPI()

    # Override ONLY the auth provider dependency - let everything else be real
    app.dependency_overrides[get_auth_provider] = lambda: mock_cognito_provider

    # Also mock the lockout service to prevent test interference
    mock_lockout = AsyncMock()  # Use AsyncMock for async methods
    mock_lockout.check_account.return_value = None  # No lockout
    mock_lockout.check_lockout.return_value = None  # No lockout
    mock_lockout.record_failed_attempt.return_value = None
    mock_lockout.reset_attempts.return_value = None
    mock_lockout.is_locked.return_value = False
    app.dependency_overrides[get_lockout_service] = lambda: mock_lockout

    # Include the REAL auth router
    app.include_router(router, prefix="/api/v1/auth")

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client with real app."""
    return TestClient(app)


class TestUserRegistration:
    """Test user registration endpoint with real code."""

    @pytest.mark.asyncio
    async def test_register_success(
        self,
        client: TestClient,
        mock_cognito_provider: Mock,
        test_user: User,
        auth_tokens: dict[str, Any],
    ) -> None:
        """Test successful user registration."""
        mock_cognito_provider.create_user.return_value = test_user
        mock_cognito_provider.authenticate.return_value = auth_tokens

        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "SecurePass123!",
                "display_name": "New User",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == auth_tokens["access_token"]
        assert data["refresh_token"] == auth_tokens["refresh_token"]
        assert data["token_type"] == AUTH_HEADER_TYPE_BEARER
        assert data["expires_in"] == 3600
        assert data["scope"] == AUTH_SCOPE_FULL_ACCESS

    @pytest.mark.asyncio
    async def test_register_user_already_exists(
        self, client: TestClient, mock_cognito_provider: Mock
    ) -> None:
        """Test registration when user already exists."""
        mock_cognito_provider.create_user.side_effect = UserAlreadyExistsError(
            "User already exists"
        )

        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "existing@example.com",
                "password": "Password123!",
            },
        )

        assert response.status_code == 409
        assert "User already exists" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_register_validation_error(self, client: TestClient) -> None:
        """Test registration with invalid data."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "invalid-email",  # Invalid email format
                "password": "weak",  # Too short
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_invalid_password(
        self, client: TestClient, mock_cognito_provider: Mock
    ) -> None:
        """Test registration with password that doesn't meet Cognito policy."""
        mock_cognito_provider.create_user.side_effect = InvalidCredentialsError(
            "Password did not conform with policy: Password must have uppercase characters"
        )

        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "password123",  # No uppercase
            },
        )

        assert response.status_code == 400
        assert "Password must have uppercase characters" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_register_create_user_fails(
        self, client: TestClient, mock_cognito_provider: Mock
    ) -> None:
        """Test registration when user creation fails."""
        mock_cognito_provider.create_user.return_value = None

        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "Password123!",
            },
        )

        assert response.status_code == 500
        assert "Failed to create user" in response.json()["detail"]


class TestUserLogin:
    """Test user login endpoint with real code."""

    @pytest.mark.asyncio
    async def test_login_success(
        self,
        client: TestClient,
        mock_cognito_provider: Mock,
        auth_tokens: dict[str, Any],
    ) -> None:
        """Test successful login."""
        mock_cognito_provider.authenticate.return_value = auth_tokens

        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "Password123!",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == auth_tokens["access_token"]
        assert data["refresh_token"] == auth_tokens["refresh_token"]
        assert data["token_type"] == AUTH_HEADER_TYPE_BEARER

    @pytest.mark.asyncio
    async def test_login_invalid_credentials(
        self, client: TestClient, mock_cognito_provider: Mock
    ) -> None:
        """Test login with invalid credentials."""
        mock_cognito_provider.authenticate.side_effect = InvalidCredentialsError(
            "Invalid email or password"
        )

        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "WrongPassword",
            },
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_login_email_not_verified(
        self, client: TestClient, mock_cognito_provider: Mock
    ) -> None:
        """Test login with unverified email."""
        mock_cognito_provider.authenticate.side_effect = EmailNotVerifiedError(
            "Email not verified"
        )

        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "Password123!",
            },
        )

        assert response.status_code == 403


class TestCurrentUser:
    """Test current user endpoint with real code."""

    def test_get_current_user_success(self, app: FastAPI) -> None:
        """Test successful get current user."""
        current_user = {
            "uid": "user-123",
            "email": "test@example.com",
            "email_verified": True,
            "display_name": "Test User",
            "auth_provider": "cognito",
        }

        # Override just the get_current_user dependency
        from clarity.auth.dependencies import get_current_user  # noqa: PLC0415

        app.dependency_overrides[get_current_user] = lambda: current_user

        client = TestClient(app)
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer test-token"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user-123"
        assert data["email"] == "test@example.com"


class TestUpdateUser:
    """Test update user endpoint with real code."""

    @pytest.mark.asyncio
    async def test_update_user_success(
        self, app: FastAPI, mock_cognito_provider: Mock, test_user: User
    ) -> None:
        """Test successful user update."""
        current_user = {
            "uid": test_user.uid,
            "email": test_user.email,
        }

        updated_user = User(
            uid=test_user.uid,
            email=test_user.email,
            display_name="Updated Name",
        )
        mock_cognito_provider.update_user.return_value = updated_user

        # Override just the get_current_user dependency
        from clarity.auth.dependencies import get_current_user  # noqa: PLC0415

        app.dependency_overrides[get_current_user] = lambda: current_user

        client = TestClient(app)
        response = client.put(
            "/api/v1/auth/me",
            json={"display_name": "Updated Name"},
            headers={"Authorization": "Bearer test-token"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == test_user.uid
        assert data["display_name"] == "Updated Name"


class TestLogout:
    """Test logout endpoint with real code."""

    def test_logout_success(self, client: TestClient) -> None:
        """Test successful logout."""
        with patch("clarity.api.v1.auth.get_user_func") as mock_get_user:
            mock_get_user.return_value = {"uid": "user-123"}

            response = client.post(
                "/api/v1/auth/logout",
                headers={"Authorization": "Bearer test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Successfully logged out"

    def test_logout_no_auth(self, client: TestClient) -> None:
        """Test logout without auth."""
        response = client.post("/api/v1/auth/logout")

        assert response.status_code == 422


class TestHealthCheck:
    """Test health check endpoint with real code."""

    def test_health_check_success(self, client: TestClient) -> None:
        """Test successful health check."""
        response = client.get("/api/v1/auth/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "authentication"


class TestRefreshToken:
    """Test refresh token endpoint with real code."""

    @pytest.mark.asyncio
    async def test_refresh_token_success(
        self, client: TestClient, mock_cognito_provider: Mock
    ) -> None:
        """Test successful token refresh."""
        mock_cognito_provider.cognito_client.initiate_auth.return_value = {
            "AuthenticationResult": {
                "AccessToken": "new-access-token",
                "ExpiresIn": 3600,
            }
        }

        response = client.post(
            "/api/v1/auth/refresh",
            headers={"Authorization": "Bearer test-refresh-token"},
        )

        assert response.status_code == 200
        data = response.json()
        assert (
            data["access_token"] == "new-access-token"  # noqa: S105 - Test token value
        )
        assert data["expires_in"] == 3600

    def test_refresh_token_missing(self, client: TestClient) -> None:
        """Test refresh without token."""
        response = client.post("/api/v1/auth/refresh")

        assert response.status_code == 422
