"""Tests for authentication middleware."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.requests import Request
from starlette.responses import Response

from clarity.middleware.auth_middleware import CognitoAuthMiddleware
from clarity.models.auth import Permission, UserContext, UserRole


@pytest.fixture
def mock_app():
    """Create a mock ASGI application."""
    return MagicMock()


@pytest.fixture
def mock_request():
    """Create a mock request."""
    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.headers = {}
    request.url = MagicMock()
    request.url.path = "/api/v1/health-data"
    return request


@pytest.fixture
def mock_user_context():
    """Create a mock user context."""
    return UserContext(
        user_id="test-user-123",
        email="test@example.com",
        role=UserRole.PATIENT,
        permissions=[Permission.READ_OWN_DATA, Permission.WRITE_OWN_DATA],
        is_verified=True,
        is_active=True,
        custom_claims={},
    )


class TestCognitoAuthMiddleware:
    """Test cases for Cognito authentication middleware."""

    @pytest.mark.asyncio
    async def test_public_path_bypass(self, mock_app: Any, mock_request: Any) -> None:
        """Test that public paths bypass authentication."""
        # Arrange
        mock_request.url.path = "/health"
        call_next = AsyncMock(return_value=Response("OK"))

        middleware = CognitoAuthMiddleware(mock_app)

        # Act
        response = await middleware.dispatch(mock_request, call_next)

        # Assert
        assert response.body == b"OK"
        assert mock_request.state.user is None
        call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_no_auth_header(self, mock_app: Any, mock_request: Any) -> None:
        """Test request without Authorization header."""
        # Arrange
        call_next = AsyncMock(return_value=Response("OK"))

        with patch.dict("os.environ", {"ENABLE_AUTH": "true"}):
            middleware = CognitoAuthMiddleware(mock_app)

        # Act
        response = await middleware.dispatch(mock_request, call_next)

        # Assert
        assert response.body == b"OK"
        assert mock_request.state.user is None
        call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_invalid_auth_header_format(
        self, mock_app: Any, mock_request: Any
    ) -> None:
        """Test request with invalid Authorization header format."""
        # Arrange
        mock_request.headers = {"Authorization": "Invalid token"}
        call_next = AsyncMock(return_value=Response("OK"))

        with patch.dict("os.environ", {"ENABLE_AUTH": "true"}):
            middleware = CognitoAuthMiddleware(mock_app)

        # Act
        response = await middleware.dispatch(mock_request, call_next)

        # Assert
        assert response.body == b"OK"
        assert mock_request.state.user is None
        call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_auth_disabled(self, mock_app: Any, mock_request: Any) -> None:
        """Test when authentication is disabled."""
        # Arrange
        mock_request.headers = {"Authorization": "Bearer test-token"}
        call_next = AsyncMock(return_value=Response("OK"))

        with patch.dict("os.environ", {"ENABLE_AUTH": "false"}):
            middleware = CognitoAuthMiddleware(mock_app)

        # Act
        response = await middleware.dispatch(mock_request, call_next)

        # Assert
        assert response.body == b"OK"
        assert mock_request.state.user is None
        call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_valid_token_authentication(
        self, mock_app: Any, mock_request: Any, mock_user_context: UserContext
    ) -> None:
        """Test successful authentication with valid token."""
        # Arrange
        mock_request.headers = {"Authorization": "Bearer valid-token"}
        call_next = AsyncMock(return_value=Response("OK"))

        with patch.dict(
            "os.environ",
            {
                "ENABLE_AUTH": "true",
                "COGNITO_USER_POOL_ID": "test-pool",
                "COGNITO_CLIENT_ID": "test-client",
            },
        ):
            middleware = CognitoAuthMiddleware(mock_app)

            # Mock the auth provider
            mock_auth_provider = MagicMock()
            mock_auth_provider._initialized = True
            mock_auth_provider.verify_token = AsyncMock(
                return_value={
                    "user_id": "test-user-123",
                    "email": "test@example.com",
                    "verified": True,
                }
            )
            mock_auth_provider.get_or_create_user_context = AsyncMock(
                return_value=mock_user_context
            )

            middleware.auth_provider = mock_auth_provider

        # Act
        with patch(
            "clarity.middleware.auth_middleware.set_user_context"
        ) as mock_set_context:
            response = await middleware.dispatch(mock_request, call_next)

        # Assert
        assert response.body == b"OK"
        assert mock_request.state.user == mock_user_context
        mock_auth_provider.verify_token.assert_called_once_with("valid-token")
        mock_set_context.assert_called_once_with(mock_user_context)
        call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_invalid_token_authentication(
        self, mock_app: Any, mock_request: Any
    ) -> None:
        """Test authentication with invalid token."""
        # Arrange
        mock_request.headers = {"Authorization": "Bearer invalid-token"}
        call_next = AsyncMock(return_value=Response("OK"))

        with patch.dict(
            "os.environ",
            {
                "ENABLE_AUTH": "true",
                "COGNITO_USER_POOL_ID": "test-pool",
                "COGNITO_CLIENT_ID": "test-client",
            },
        ):
            middleware = CognitoAuthMiddleware(mock_app)

            # Mock the auth provider
            mock_auth_provider = MagicMock()
            mock_auth_provider._initialized = True
            mock_auth_provider.verify_token = AsyncMock(
                side_effect=Exception("Invalid token")
            )

            middleware.auth_provider = mock_auth_provider

        # Act
        response = await middleware.dispatch(mock_request, call_next)

        # Assert
        assert response.body == b"OK"
        assert mock_request.state.user is None
        mock_auth_provider.verify_token.assert_called_once_with("invalid-token")
        call_next.assert_called_once_with(mock_request)
