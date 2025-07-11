"""Test rate limiting middleware."""

import time
from unittest.mock import AsyncMock, Mock, patch

from fastapi import Request, Response
import pytest
from starlette.datastructures import Headers

from clarity.middleware.rate_limiting import (
    RateLimitingMiddleware,
    get_ip_only,
    get_user_id_or_ip,
)


class TestRateLimitHelpers:
    """Test rate limiting helper functions."""

    def test_get_user_id_or_ip_with_user(self):
        """Test user ID extraction with authenticated user."""
        mock_request = Mock(spec=Request)
        mock_request.state.user = {"user_id": "user123"}

        result = get_user_id_or_ip(mock_request)
        assert result == "user123"

    def test_get_user_id_or_ip_with_uid(self):
        """Test user ID extraction with uid field."""
        mock_request = Mock(spec=Request)
        mock_request.state.user = {"uid": "uid456"}

        result = get_user_id_or_ip(mock_request)
        assert result == "uid456"

    def test_get_user_id_or_ip_fallback_to_ip(self):
        """Test fallback to IP when no user."""
        mock_request = Mock(spec=Request)
        mock_request.state.user = None
        mock_request.client = Mock(host="192.168.1.1")

        result = get_user_id_or_ip(mock_request)
        assert result == "192.168.1.1"

    def test_get_ip_only_with_client(self):
        """Test IP extraction with client."""
        mock_request = Mock(spec=Request)
        mock_request.client = Mock(host="10.0.0.1")

        result = get_ip_only(mock_request)
        assert result == "10.0.0.1"

    def test_get_ip_only_no_client(self):
        """Test IP extraction without client."""
        mock_request = Mock(spec=Request)
        mock_request.client = None

        result = get_ip_only(mock_request)
        assert result == "127.0.0.1"


class TestRateLimitingMiddleware:
    """Test RateLimitingMiddleware class."""

    @pytest.mark.asyncio
    async def test_middleware_initialization(self):
        """Test middleware initialization."""
        mock_limiter = Mock()
        middleware = RateLimitingMiddleware(
            app=None, limiter=mock_limiter, key_func=get_user_id_or_ip
        )

        assert middleware.limiter == mock_limiter
        assert middleware.key_func == get_user_id_or_ip

    @pytest.mark.asyncio
    async def test_middleware_dispatch_allowed(self):
        """Test middleware dispatch when request is allowed."""
        mock_limiter = Mock()
        mock_limiter.hit = AsyncMock(return_value={"headers": {}})

        middleware = RateLimitingMiddleware(
            app=None, limiter=mock_limiter, key_func=get_user_id_or_ip
        )

        mock_request = Mock(spec=Request)
        mock_request.state.user = {"user_id": "test"}
        mock_request.url.path = "/api/test"

        mock_response = Mock(spec=Response)
        mock_response.headers = {}

        async def mock_call_next(request):
            return mock_response

        result = await middleware.dispatch(mock_request, mock_call_next)

        assert result == mock_response
        mock_limiter.hit.assert_called_once()
