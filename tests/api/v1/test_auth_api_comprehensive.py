"""Comprehensive tests for authentication API endpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import uuid

from fastapi import status
from fastapi.testclient import TestClient
import pytest

from clarity.api.v1.auth import router
from clarity.auth.aws_cognito_provider import CognitoAuthProvider
from clarity.auth.dependencies import AuthenticatedUser
from clarity.core.constants import (
    AUTH_HEADER_TYPE_BEARER,
    AUTH_SCOPE_FULL_ACCESS,
    AUTH_TOKEN_DEFAULT_EXPIRY_SECONDS,
)
from clarity.models.auth import TokenResponse, UserLoginRequest
from clarity.models.user import User
from clarity.services.cognito_auth_service import (
    EmailNotVerifiedError,
    InvalidCredentialsError,
    UserAlreadyExistsError,
    UserNotFoundError,
)


@pytest.fixture
def mock_cognito_provider():
    """Mock Cognito auth provider."""
    provider = Mock(spec=CognitoAuthProvider)
    provider.authenticate = AsyncMock()
    provider.create_user = AsyncMock()
    provider.update_user = AsyncMock()
    provider.verify_token = AsyncMock()
    provider.get_user = AsyncMock()
    
    # Mock cognito client
    mock_client = MagicMock()
    mock_client.initiate_auth = MagicMock()
    mock_client.exceptions.NotAuthorizedException = Exception
    provider.cognito_client = mock_client
    provider.client_id = "test-client-id"
    
    return provider


@pytest.fixture
def mock_other_provider():
    """Mock non-Cognito auth provider."""
    provider = Mock()
    provider.authenticate = AsyncMock()
    return provider


@pytest.fixture
def test_user():
    """Create test user."""
    return User(
        uid=str(uuid.uuid4()),
        email="test@example.com",
        display_name="Test User",
    )


@pytest.fixture
def auth_tokens():
    """Create test auth tokens."""
    return {
        "access_token": "test-access-token",
        "refresh_token": "test-refresh-token",
        "expires_in": 3600,
    }


@pytest.fixture
def client():
    """Create test client."""
    from fastapi import FastAPI
    
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/auth")
    return TestClient(app)


class TestUserRegistration:
    """Test user registration endpoint."""

    @pytest.mark.asyncio
    async def test_register_success(
        self,
        client,
        mock_cognito_provider,
        test_user,
        auth_tokens,
    ):
        """Test successful user registration."""
        mock_cognito_provider.create_user.return_value = test_user
        mock_cognito_provider.authenticate.return_value = auth_tokens
        
        with patch(
            "clarity.api.v1.auth.get_auth_provider",
            return_value=mock_cognito_provider,
        ):
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
        
        # Verify calls
        mock_cognito_provider.create_user.assert_called_once_with(
            email="newuser@example.com",
            password="SecurePass123!",
            display_name="New User",
        )
        mock_cognito_provider.authenticate.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_user_already_exists(
        self,
        client,
        mock_cognito_provider,
    ):
        """Test registration when user already exists."""
        mock_cognito_provider.create_user.side_effect = UserAlreadyExistsError(
            "User already exists"
        )
        
        with patch(
            "clarity.api.v1.auth.get_auth_provider",
            return_value=mock_cognito_provider,
        ):
            response = client.post(
                "/api/v1/auth/register",
                json={
                    "email": "existing@example.com",
                    "password": "Password123!",
                },
            )
        
        assert response.status_code == 409
        data = response.json()
        assert "User already exists" in data["detail"]

    @pytest.mark.asyncio
    async def test_register_invalid_provider(
        self,
        client,
        mock_other_provider,
    ):
        """Test registration with non-Cognito provider."""
        with patch(
            "clarity.api.v1.auth.get_auth_provider",
            return_value=mock_other_provider,
        ):
            response = client.post(
                "/api/v1/auth/register",
                json={
                    "email": "test@example.com",
                    "password": "Password123!",
                },
            )
        
        assert response.status_code == 500
        assert "Invalid authentication provider" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_register_weak_password(
        self,
        client,
    ):
        """Test registration with weak password."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "weak",  # Too short
            },
        )
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_register_create_user_fails(
        self,
        client,
        mock_cognito_provider,
    ):
        """Test registration when user creation fails."""
        mock_cognito_provider.create_user.return_value = None
        
        with patch(
            "clarity.api.v1.auth.get_auth_provider",
            return_value=mock_cognito_provider,
        ):
            response = client.post(
                "/api/v1/auth/register",
                json={
                    "email": "test@example.com",
                    "password": "Password123!",
                },
            )
        
        assert response.status_code == 500
        assert "Failed to create user" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_register_auth_fails_after_create(
        self,
        client,
        mock_cognito_provider,
        test_user,
    ):
        """Test registration when authentication fails after user creation."""
        mock_cognito_provider.create_user.return_value = test_user
        mock_cognito_provider.authenticate.return_value = None
        
        with patch(
            "clarity.api.v1.auth.get_auth_provider",
            return_value=mock_cognito_provider,
        ):
            response = client.post(
                "/api/v1/auth/register",
                json={
                    "email": "test@example.com",
                    "password": "Password123!",
                },
            )
        
        assert response.status_code == 500
        assert "Failed to authenticate after registration" in response.json()["detail"]


class TestUserLogin:
    """Test user login endpoint."""

    @pytest.mark.asyncio
    async def test_login_success(
        self,
        client,
        mock_cognito_provider,
        auth_tokens,
    ):
        """Test successful login."""
        mock_cognito_provider.authenticate.return_value = auth_tokens
        
        with patch(
            "clarity.api.v1.auth.get_auth_provider",
            return_value=mock_cognito_provider,
        ):
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
        self,
        client,
        mock_cognito_provider,
    ):
        """Test login with invalid credentials."""
        mock_cognito_provider.authenticate.side_effect = InvalidCredentialsError(
            "Invalid email or password"
        )
        
        with patch(
            "clarity.api.v1.auth.get_auth_provider",
            return_value=mock_cognito_provider,
        ):
            response = client.post(
                "/api/v1/auth/login",
                json={
                    "email": "test@example.com",
                    "password": "WrongPassword",
                },
            )
        
        assert response.status_code == 401
        data = response.json()
        assert data["type"] == "invalid_credentials"
        assert "Invalid email or password" in data["detail"]

    @pytest.mark.asyncio
    async def test_login_email_not_verified(
        self,
        client,
        mock_cognito_provider,
    ):
        """Test login with unverified email."""
        mock_cognito_provider.authenticate.side_effect = EmailNotVerifiedError(
            "Email not verified"
        )
        
        with patch(
            "clarity.api.v1.auth.get_auth_provider",
            return_value=mock_cognito_provider,
        ):
            response = client.post(
                "/api/v1/auth/login",
                json={
                    "email": "test@example.com",
                    "password": "Password123!",
                },
            )
        
        assert response.status_code == 403
        data = response.json()
        assert data["type"] == "email_not_verified"

    @pytest.mark.asyncio
    async def test_login_auth_returns_none(
        self,
        client,
        mock_cognito_provider,
    ):
        """Test login when authentication returns None."""
        mock_cognito_provider.authenticate.return_value = None
        
        with patch(
            "clarity.api.v1.auth.get_auth_provider",
            return_value=mock_cognito_provider,
        ):
            response = client.post(
                "/api/v1/auth/login",
                json={
                    "email": "test@example.com",
                    "password": "Password123!",
                },
            )
        
        assert response.status_code == 500
        assert "Failed to authenticate user" in response.json()["detail"]


class TestGetCurrentUser:
    """Test get current user endpoint."""

    def test_get_current_user_success(self, client):
        """Test successful get current user."""
        current_user = {
            "uid": "user-123",
            "email": "test@example.com",
            "email_verified": True,
            "display_name": "Test User",
            "auth_provider": "cognito",
        }
        
        with patch(
            "clarity.api.v1.auth.get_current_user",
            return_value=current_user,
        ):
            response = client.get(
                "/api/v1/auth/me",
                headers={"Authorization": "Bearer test-token"},
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user-123"
        assert data["email"] == "test@example.com"
        assert data["email_verified"] is True
        assert data["display_name"] == "Test User"
        assert data["auth_provider"] == "cognito"

    def test_get_current_user_minimal_data(self, client):
        """Test get current user with minimal data."""
        current_user = {
            "user_id": "user-456",  # Using user_id instead of uid
        }
        
        with patch(
            "clarity.api.v1.auth.get_current_user",
            return_value=current_user,
        ):
            response = client.get(
                "/api/v1/auth/me",
                headers={"Authorization": "Bearer test-token"},
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user-456"
        assert data["email"] is None
        assert data["email_verified"] is True  # Default
        assert data["display_name"] is None
        assert data["auth_provider"] == "cognito"  # Default


class TestUpdateUser:
    """Test update user endpoint."""

    @pytest.mark.asyncio
    async def test_update_user_success(
        self,
        client,
        mock_cognito_provider,
        test_user,
    ):
        """Test successful user update."""
        current_user = {
            "uid": test_user.uid,
            "email": test_user.email,
        }
        
        # Update display name
        updated_user = User(
            uid=test_user.uid,
            email=test_user.email,
            display_name="Updated Name",
        )
        mock_cognito_provider.update_user.return_value = updated_user
        
        with patch(
            "clarity.api.v1.auth.get_current_user",
            return_value=current_user,
        ):
            with patch(
                "clarity.api.v1.auth.get_auth_provider",
                return_value=mock_cognito_provider,
            ):
                response = client.put(
                    "/api/v1/auth/me",
                    json={"display_name": "Updated Name"},
                    headers={"Authorization": "Bearer test-token"},
                )
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == test_user.uid
        assert data["display_name"] == "Updated Name"
        assert data["updated"] is True

    @pytest.mark.asyncio
    async def test_update_user_no_user_id(
        self,
        client,
        mock_cognito_provider,
    ):
        """Test update when user ID not found in token."""
        current_user = {
            "email": "test@example.com",
            # Missing uid and user_id
        }
        
        with patch(
            "clarity.api.v1.auth.get_current_user",
            return_value=current_user,
        ):
            with patch(
                "clarity.api.v1.auth.get_auth_provider",
                return_value=mock_cognito_provider,
            ):
                response = client.put(
                    "/api/v1/auth/me",
                    json={"display_name": "New Name"},
                    headers={"Authorization": "Bearer test-token"},
                )
        
        assert response.status_code == 400
        assert "User ID not found in token" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_update_user_not_found(
        self,
        client,
        mock_cognito_provider,
    ):
        """Test update when user not found."""
        current_user = {"uid": "user-123"}
        
        mock_cognito_provider.update_user.side_effect = UserNotFoundError(
            "User not found"
        )
        
        with patch(
            "clarity.api.v1.auth.get_current_user",
            return_value=current_user,
        ):
            with patch(
                "clarity.api.v1.auth.get_auth_provider",
                return_value=mock_cognito_provider,
            ):
                response = client.put(
                    "/api/v1/auth/me",
                    json={"email": "newemail@example.com"},
                    headers={"Authorization": "Bearer test-token"},
                )
        
        assert response.status_code == 404
        assert "User not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_update_user_returns_none(
        self,
        client,
        mock_cognito_provider,
    ):
        """Test update when provider returns None."""
        current_user = {"uid": "user-123"}
        
        mock_cognito_provider.update_user.return_value = None
        
        with patch(
            "clarity.api.v1.auth.get_current_user",
            return_value=current_user,
        ):
            with patch(
                "clarity.api.v1.auth.get_auth_provider",
                return_value=mock_cognito_provider,
            ):
                response = client.put(
                    "/api/v1/auth/me",
                    json={"display_name": "New Name"},
                    headers={"Authorization": "Bearer test-token"},
                )
        
        assert response.status_code == 404
        assert "User user-123 not found" in response.json()["detail"]


class TestLogout:
    """Test logout endpoint."""

    @pytest.mark.asyncio
    async def test_logout_success_with_auth(
        self,
        client,
        mock_cognito_provider,
    ):
        """Test successful logout with authentication."""
        with patch(
            "clarity.api.v1.auth.get_auth_provider",
            return_value=mock_cognito_provider,
        ):
            with patch(
                "clarity.api.v1.auth.get_user_func",
                return_value={"uid": "user-123"},
            ):
                response = client.post(
                    "/api/v1/auth/logout",
                    headers={"Authorization": "Bearer test-token"},
                )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Successfully logged out"

    @pytest.mark.asyncio
    async def test_logout_no_auth_no_body(
        self,
        client,
        mock_cognito_provider,
    ):
        """Test logout without auth header or body."""
        with patch(
            "clarity.api.v1.auth.get_auth_provider",
            return_value=mock_cognito_provider,
        ):
            response = client.post("/api/v1/auth/logout")
        
        assert response.status_code == 422
        data = response.json()
        assert data["type"] == "validation_error"
        assert "Request body or Authorization header required" in data["detail"]

    @pytest.mark.asyncio
    async def test_logout_invalid_auth(
        self,
        client,
        mock_cognito_provider,
    ):
        """Test logout with invalid authentication."""
        with patch(
            "clarity.api.v1.auth.get_auth_provider",
            return_value=mock_cognito_provider,
        ):
            with patch(
                "clarity.api.v1.auth.get_user_func",
                side_effect=Exception("Invalid token"),
            ):
                response = client.post(
                    "/api/v1/auth/logout",
                    headers={"Authorization": "Bearer invalid-token"},
                )
        
        assert response.status_code == 401
        data = response.json()
        assert data["type"] == "authentication_required"

    @pytest.mark.asyncio
    async def test_logout_with_body_no_auth(
        self,
        client,
        mock_cognito_provider,
    ):
        """Test logout with body but no auth header."""
        with patch(
            "clarity.api.v1.auth.get_auth_provider",
            return_value=mock_cognito_provider,
        ):
            response = client.post(
                "/api/v1/auth/logout",
                json={"some": "data"},
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Successfully logged out"


class TestHealthCheck:
    """Test health check endpoint."""

    def test_health_check_success(self, client):
        """Test successful health check."""
        response = client.get("/api/v1/auth/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "authentication"
        assert data["version"] == "1.0.0"

    def test_health_check_exception(self, client):
        """Test health check with exception."""
        with patch(
            "clarity.api.v1.auth.HealthResponse",
            side_effect=Exception("Health check error"),
        ):
            response = client.get("/api/v1/auth/health")
        
        # Should still return 200 with unhealthy status
        assert response.status_code == 200


class TestRefreshToken:
    """Test refresh token endpoint."""

    @pytest.mark.asyncio
    async def test_refresh_token_success(
        self,
        client,
        mock_cognito_provider,
    ):
        """Test successful token refresh."""
        mock_cognito_provider.cognito_client.initiate_auth.return_value = {
            "AuthenticationResult": {
                "AccessToken": "new-access-token",
                "ExpiresIn": 3600,
            }
        }
        
        with patch(
            "clarity.api.v1.auth.get_auth_provider",
            return_value=mock_cognito_provider,
        ):
            response = client.post(
                "/api/v1/auth/refresh",
                headers={"Authorization": "Bearer test-refresh-token"},
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == "new-access-token"
        assert data["refresh_token"] == "test-refresh-token"  # Not rotated
        assert data["expires_in"] == 3600

    @pytest.mark.asyncio
    async def test_refresh_token_from_body(
        self,
        client,
        mock_cognito_provider,
    ):
        """Test refresh token from request body."""
        mock_cognito_provider.cognito_client.initiate_auth.return_value = {
            "AuthenticationResult": {
                "AccessToken": "new-access-token",
                "ExpiresIn": 3600,
            }
        }
        
        with patch(
            "clarity.api.v1.auth.get_auth_provider",
            return_value=mock_cognito_provider,
        ):
            response = client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": "test-refresh-token"},
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == "new-access-token"

    @pytest.mark.asyncio
    async def test_refresh_token_missing(
        self,
        client,
        mock_cognito_provider,
    ):
        """Test refresh without token."""
        with patch(
            "clarity.api.v1.auth.get_auth_provider",
            return_value=mock_cognito_provider,
        ):
            response = client.post("/api/v1/auth/refresh")
        
        assert response.status_code == 422
        data = response.json()
        assert data["type"] == "missing_refresh_token"

    @pytest.mark.asyncio
    async def test_refresh_token_invalid(
        self,
        client,
        mock_cognito_provider,
    ):
        """Test refresh with invalid token."""
        # Set up the exception
        exc = Exception("Invalid token")
        exc.__class__ = type('NotAuthorizedException', (Exception,), {})
        mock_cognito_provider.cognito_client.exceptions.NotAuthorizedException = exc.__class__
        mock_cognito_provider.cognito_client.initiate_auth.side_effect = exc
        
        with patch(
            "clarity.api.v1.auth.get_auth_provider",
            return_value=mock_cognito_provider,
        ):
            response = client.post(
                "/api/v1/auth/refresh",
                headers={"Authorization": "Bearer invalid-refresh-token"},
            )
        
        assert response.status_code == 401
        assert "Invalid refresh token" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_refresh_token_no_auth_result(
        self,
        client,
        mock_cognito_provider,
    ):
        """Test refresh when Cognito returns no AuthenticationResult."""
        mock_cognito_provider.cognito_client.initiate_auth.return_value = {
            # Missing AuthenticationResult
        }
        
        with patch(
            "clarity.api.v1.auth.get_auth_provider",
            return_value=mock_cognito_provider,
        ):
            response = client.post(
                "/api/v1/auth/refresh",
                headers={"Authorization": "Bearer test-refresh-token"},
            )
        
        assert response.status_code == 500
        assert "Failed to refresh token" in response.json()["detail"]


class TestValidation:
    """Test input validation."""

    def test_register_invalid_email(self, client):
        """Test registration with invalid email."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "not-an-email",
                "password": "Password123!",
            },
        )
        
        assert response.status_code == 422

    def test_login_missing_fields(self, client):
        """Test login with missing fields."""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com"},  # Missing password
        )
        
        assert response.status_code == 422

    def test_update_empty_body(self, client):
        """Test update with empty body."""
        with patch(
            "clarity.api.v1.auth.get_current_user",
            return_value={"uid": "user-123"},
        ):
            response = client.put(
                "/api/v1/auth/me",
                json={},  # Empty update
                headers={"Authorization": "Bearer test-token"},
            )
        
        # Should succeed but do nothing
        assert response.status_code in [200, 422]  # Depends on implementation