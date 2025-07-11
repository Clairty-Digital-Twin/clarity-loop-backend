"""Fast unit tests for auth.py to boost coverage.

Focus on testing simple functions and error paths.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
import pytest

from clarity.api.v1.auth import (
    EmailConfirmationRequest,
    ForgotPasswordRequest,
    HealthResponse,
    LogoutResponse,
    ResendConfirmationRequest,
    ResetPasswordRequest,
    StatusResponse,
    UserInfoResponse,
    UserRegister,
    UserUpdate,
    UserUpdateResponse,
    auth_health,
    confirm_email,
    forgot_password,
    get_current_user_info,
    logout,
    refresh_token,
    register,
    resend_confirmation,
    reset_password,
    update_user,
)
from clarity.auth.aws_cognito_provider import CognitoAuthProvider


class TestAuthModels:
    """Test auth request/response models."""

    def test_user_register_model(self):
        """Test UserRegister model validation."""
        user = UserRegister(
            email="test@example.com", password="securepass123", display_name="Test User"
        )
        assert user.email == "test@example.com"
        assert user.password == "securepass123"
        assert user.display_name == "Test User"

    def test_user_info_response_model(self):
        """Test UserInfoResponse model."""
        response = UserInfoResponse(
            user_id="123",
            email="test@example.com",
            email_verified=True,
            display_name="Test User",
            auth_provider="cognito",
        )
        assert response.user_id == "123"
        assert response.email_verified is True

    def test_status_response_model(self):
        """Test StatusResponse model."""
        response = StatusResponse(status="confirmed")
        assert response.status == "confirmed"

    def test_health_response_model(self):
        """Test HealthResponse model."""
        response = HealthResponse(
            status="healthy", service="authentication", version="1.0.0"
        )
        assert response.status == "healthy"
        assert response.service == "authentication"


class TestAuthEndpoints:
    """Test auth endpoint functions."""

    @pytest.mark.asyncio
    async def test_auth_health_endpoint(self):
        """Test auth health check endpoint."""
        response = await auth_health()
        assert response.status == "healthy"
        assert response.service == "authentication"
        assert response.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_get_current_user_info_basic(self):
        """Test get current user info with basic data."""
        mock_user = {
            "uid": "user123",
            "email": "test@example.com",
            "email_verified": True,
            "display_name": "Test User",
            "auth_provider": "cognito",
        }

        response = await get_current_user_info(current_user=mock_user)
        assert response.user_id == "user123"
        assert response.email == "test@example.com"
        assert response.email_verified is True

    @pytest.mark.asyncio
    async def test_get_current_user_info_fallback_user_id(self):
        """Test get current user info with user_id fallback."""
        mock_user = {
            "user_id": "user456",  # No uid, should fallback to user_id
            "email": "test@example.com",
        }

        response = await get_current_user_info(current_user=mock_user)
        assert response.user_id == "user456"

    @pytest.mark.asyncio
    async def test_update_user_invalid_provider(self):
        """Test update user with invalid auth provider."""
        mock_user = {"uid": "user123"}
        mock_provider = Mock()  # Not a CognitoAuthProvider

        with pytest.raises(HTTPException) as exc_info:
            await update_user(
                updates=UserUpdate(display_name="New Name"),
                current_user=mock_user,
                auth_provider=mock_provider,
            )

        assert exc_info.value.status_code == 500
        assert "Invalid authentication provider" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_update_user_no_user_id(self):
        """Test update user with missing user ID."""
        mock_user = {}  # No uid or user_id
        mock_provider = Mock(spec=CognitoAuthProvider)

        with pytest.raises(HTTPException) as exc_info:
            await update_user(
                updates=UserUpdate(display_name="New Name"),
                current_user=mock_user,
                auth_provider=mock_provider,
            )

        assert exc_info.value.status_code == 400
        assert "User ID not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_logout_empty_body_and_no_auth(self):
        """Test logout with empty body and no auth header."""
        mock_request = Mock()
        mock_request.headers.get.return_value = ""  # No auth header
        mock_request.json = AsyncMock(side_effect=ValueError)  # Empty body
        mock_provider = Mock()

        with pytest.raises(HTTPException) as exc_info:
            await logout(request=mock_request, _auth_provider=mock_provider)

        assert exc_info.value.status_code == 422
        assert "Request body or Authorization header required" in str(
            exc_info.value.detail["detail"]
        )

    @pytest.mark.asyncio
    async def test_confirm_email_invalid_provider(self):
        """Test confirm email with invalid auth provider."""
        mock_provider = Mock()  # Not a CognitoAuthProvider
        request = EmailConfirmationRequest(email="test@example.com", code="123456")

        with pytest.raises(HTTPException) as exc_info:
            await confirm_email(request=request, auth_provider=mock_provider)

        assert exc_info.value.status_code == 500
        assert "Invalid authentication provider" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_resend_confirmation_invalid_provider(self):
        """Test resend confirmation with invalid auth provider."""
        mock_provider = Mock()  # Not a CognitoAuthProvider
        request = ResendConfirmationRequest(email="test@example.com")

        with pytest.raises(HTTPException) as exc_info:
            await resend_confirmation(request=request, auth_provider=mock_provider)

        assert exc_info.value.status_code == 500
        assert "Invalid authentication provider" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_forgot_password_invalid_provider(self):
        """Test forgot password with invalid auth provider."""
        mock_provider = Mock()  # Not a CognitoAuthProvider
        request = ForgotPasswordRequest(email="test@example.com")

        with pytest.raises(HTTPException) as exc_info:
            await forgot_password(request=request, auth_provider=mock_provider)

        assert exc_info.value.status_code == 500
        assert "Invalid authentication provider" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_reset_password_invalid_provider(self):
        """Test reset password with invalid auth provider."""
        mock_provider = Mock()  # Not a CognitoAuthProvider
        request = ResetPasswordRequest(
            email="test@example.com", code="123456", new_password="newpass123"
        )

        with pytest.raises(HTTPException) as exc_info:
            await reset_password(request=request, auth_provider=mock_provider)

        assert exc_info.value.status_code == 500
        assert "Invalid authentication provider" in str(exc_info.value.detail)


class TestRefreshToken:
    """Test refresh token endpoint."""

    @pytest.mark.asyncio
    async def test_refresh_token_no_token(self):
        """Test refresh token with no token provided."""
        mock_request = Mock()
        mock_request.headers.get.return_value = ""
        mock_request.json = AsyncMock(return_value={})
        mock_provider = Mock(spec=CognitoAuthProvider)

        with pytest.raises(HTTPException) as exc_info:
            await refresh_token(request=mock_request, auth_provider=mock_provider)

        assert exc_info.value.status_code == 422
        assert "Refresh token is required" in str(exc_info.value.detail["detail"])

    @pytest.mark.asyncio
    async def test_refresh_token_invalid_provider(self):
        """Test refresh token with invalid auth provider."""
        mock_request = Mock()
        mock_request.headers.get.return_value = "Bearer token123"
        mock_provider = Mock()  # Not a CognitoAuthProvider

        with pytest.raises(HTTPException) as exc_info:
            await refresh_token(request=mock_request, auth_provider=mock_provider)

        assert exc_info.value.status_code == 500
        assert "Invalid authentication provider" in str(exc_info.value.detail)


class TestRegisterEndpoint:
    """Test register endpoint specific cases."""

    @pytest.mark.asyncio
    async def test_register_self_signup_disabled(self):
        """Test register when self-signup is disabled."""
        from starlette.datastructures import Headers
        from starlette.requests import Request

        # Create a minimal mock request that satisfies the rate limiter
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/register",
            "headers": [(b"host", b"localhost")],
            "query_string": b"",
            "client": ("127.0.0.1", 8000),
        }
        mock_request = Request(scope)

        user_data = UserRegister(email="test@example.com", password="password123")
        mock_provider = Mock(spec=CognitoAuthProvider)
        mock_lockout = Mock()

        with patch("clarity.api.v1.auth.os.getenv", return_value="false"):
            with pytest.raises(HTTPException) as exc_info:
                await register(
                    request=mock_request,
                    user_data=user_data,
                    auth_provider=mock_provider,
                    _lockout_service=mock_lockout,
                )

        assert exc_info.value.status_code == 403
        assert "Self-registration is currently disabled" in str(
            exc_info.value.detail["detail"]
        )


class TestAuthHappyPaths:
    """Test auth happy path scenarios."""

    def test_token_response_model(self):
        """Test TokenResponse model."""
        from clarity.models.auth import TokenResponse

        response = TokenResponse(
            access_token="mock-access-token",
            refresh_token="mock-refresh-token",
            token_type="bearer",
            expires_in=3600,
        )

        assert response.access_token == "mock-access-token"
        assert response.refresh_token == "mock-refresh-token"
        assert response.token_type == "bearer"
        assert response.expires_in == 3600

    def test_user_info_response_all_fields(self):
        """Test UserInfoResponse with all fields."""
        response = UserInfoResponse(
            user_id="123",
            email="test@example.com",
            email_verified=True,
            display_name="Test User",
            auth_provider="cognito",
            created_at="2024-01-01T00:00:00Z",
            last_login="2024-01-02T00:00:00Z",
            role="patient",
            permissions=["read", "write"],
        )
        assert response.user_id == "123"
        assert response.email == "test@example.com"
        assert response.email_verified is True
        assert response.permissions == ["read", "write"]

    def test_logout_response_model(self):
        """Test LogoutResponse model."""
        response = LogoutResponse(message="Test logout message")
        assert response.message == "Test logout message"

    def test_user_update_response_model(self):
        """Test UserUpdateResponse model."""
        response = UserUpdateResponse(
            message="Test update message", updated_fields=["field1", "field2"]
        )
        assert response.message == "Test update message"
        assert response.updated_fields == ["field1", "field2"]
