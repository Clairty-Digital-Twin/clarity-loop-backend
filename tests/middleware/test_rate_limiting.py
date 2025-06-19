"""Test suite for rate limiting middleware.

Tests various rate limiting scenarios including:
- Per-IP rate limiting
- Per-user rate limiting
- Rate limit headers
- Different endpoint limits
- Error handling
"""

import json
from typing import Any
from unittest.mock import MagicMock

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
import pytest
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from clarity.middleware.rate_limiting import (
    RateLimitingMiddleware,
    custom_rate_limit_exceeded_handler,
    get_ip_only,
    get_user_id_or_ip,
    setup_rate_limiting,
)


@pytest.fixture
def mock_request():
    """Create a mock request object."""
    request = MagicMock(spec=Request)
    request.client = MagicMock()
    request.client.host = "192.168.1.1"
    request.state = MagicMock()
    request.state.user = None
    request.url = MagicMock()
    request.url.path = "/api/test"
    request.app = MagicMock()
    request.app.state = MagicMock()
    request.app.state.limiter = MagicMock()
    request.app.state.limiter._inject_headers = lambda resp, _: resp
    request.state.view_rate_limit = None
    return request


@pytest.fixture
def authenticated_request(mock_request: Any) -> Any:
    """Create a mock authenticated request."""
    mock_request.state.user = {"uid": "user123", "email": "test@example.com"}
    return mock_request


class TestKeyFunctions:
    """Test rate limiting key extraction functions."""

    def test_get_ip_only(self, mock_request: Any) -> None:
        """Test IP-only key extraction."""
        key = get_ip_only(mock_request)
        assert key == "ip:192.168.1.1"

    def test_get_user_id_or_ip_with_user(self, authenticated_request: Any) -> None:
        """Test user ID extraction for authenticated requests."""
        key = get_user_id_or_ip(authenticated_request)
        assert key == "user:user123"

    def test_get_user_id_or_ip_without_user(self, mock_request: Any) -> None:
        """Test fallback to IP for unauthenticated requests."""
        key = get_user_id_or_ip(mock_request)
        assert key == "ip:192.168.1.1"

    def test_get_user_id_with_different_fields(self, mock_request: Any) -> None:
        """Test user ID extraction with different field names."""
        # Test with user_id field
        mock_request.state.user = {"user_id": "user456"}
        assert get_user_id_or_ip(mock_request) == "user:user456"

        # Test with sub field (JWT standard)
        mock_request.state.user = {"sub": "user789"}
        assert get_user_id_or_ip(mock_request) == "user:user789"


class TestRateLimitingMiddleware:
    """Test RateLimitingMiddleware configuration."""

    def test_create_limiter_default_config(self):
        """Test creating limiter with default configuration."""
        limiter = RateLimitingMiddleware.create_limiter()
        assert limiter is not None
        assert limiter._key_func == get_user_id_or_ip
        assert limiter._headers_enabled is True
        assert limiter._strategy == "fixed-window"

    def test_create_limiter_with_redis(self):
        """Test creating limiter with Redis storage."""
        redis_url = "redis://localhost:6379"
        limiter = RateLimitingMiddleware.create_limiter(storage_uri=redis_url)
        assert limiter is not None
        # Storage URI is set internally, we can't directly verify it

    def test_get_auth_limiter(self):
        """Test auth-specific limiter configuration."""
        limiter = RateLimitingMiddleware.get_auth_limiter()
        assert limiter is not None
        assert limiter._key_func == get_ip_only

    def test_get_ai_limiter(self):
        """Test AI endpoint limiter configuration."""
        limiter = RateLimitingMiddleware.get_ai_limiter()
        assert limiter is not None
        assert limiter._key_func == get_user_id_or_ip


class TestRateLimitExceededHandler:
    """Test custom rate limit exceeded error handler."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_handler(self, mock_request: Any) -> None:
        """Test handling of rate limit exceeded errors."""
        # Clean Code: Create a mock limit object
        mock_limit = MagicMock()
        mock_limit.limit = "5/minute"

        # Create exception with mock limit
        exc = RateLimitExceeded(mock_limit)
        exc.detail = "test_key"  # Add detail for our handler
        exc.limit = "5/minute"  # Ensure limit is accessible

        response = await custom_rate_limit_exceeded_handler(mock_request, exc)

        assert response.status_code == 429
        assert response.headers["content-type"] == "application/json"

        # Check response body
        body = json.loads(response.body)
        assert body["type"] == "rate_limit_exceeded"
        assert body["title"] == "Too Many Requests"
        assert body["status"] == 429


class TestIntegration:
    """Integration tests with FastAPI application."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI application with rate limiting."""
        app = FastAPI()

        # Set up rate limiting
        limiter = setup_rate_limiting(app)

        # Add the middleware to the app
        # Add SlowAPI middleware
        app.add_middleware(SlowAPIMiddleware)
        app.state.limiter = limiter

        # Add test endpoints
        @app.get("/test/unlimited")
        async def unlimited() -> JSONResponse:
            return JSONResponse({"message": "success"})

        @app.get("/test/limited")
        @limiter.limit("2/minute")
        async def limited(request: Request) -> JSONResponse:
            _ = request  # Used by rate limiter
            return JSONResponse({"message": "success"})

        @app.get("/test/auth")
        @limiter.limit("5/minute", key_func=get_ip_only)
        async def auth_endpoint(request: Request) -> JSONResponse:
            _ = request  # Used by rate limiter
            return JSONResponse({"message": "success"})

        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_unlimited_endpoint(self, client: TestClient) -> None:
        """Test endpoint without rate limiting."""
        for _ in range(10):
            response = client.get("/test/unlimited")
            assert response.status_code == 200

    def test_limited_endpoint_within_limit(self, client: TestClient) -> None:
        """Test rate limited endpoint within limits."""
        # First two requests should succeed
        for _ in range(2):
            response = client.get("/test/limited")
            assert response.status_code == 200
            assert "X-RateLimit-Limit" in response.headers
            assert "X-RateLimit-Remaining" in response.headers

    def test_limited_endpoint_exceeds_limit(self, client: TestClient) -> None:
        """Test rate limited endpoint exceeding limits."""
        # First two requests succeed
        for _ in range(2):
            response = client.get("/test/limited")
            assert response.status_code == 200

        # Third request should be rate limited
        response = client.get("/test/limited")
        assert response.status_code == 429
        assert response.json()["type"] == "rate_limit_exceeded"

    def test_rate_limit_headers(self, client: TestClient) -> None:
        """Test rate limit headers in responses."""
        response = client.get("/test/limited")
        assert response.status_code == 200

        # Check headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert response.headers["X-RateLimit-Limit"] == "2"
        assert response.headers["X-RateLimit-Remaining"] == "1"


class TestConfiguration:
    """Test rate limiting configuration options."""

    def test_default_limits(self):
        """Test default rate limit values."""
        assert RateLimitingMiddleware.DEFAULT_LIMITS["global"] == "1000/hour"
        assert RateLimitingMiddleware.DEFAULT_LIMITS["auth"] == "20/hour"
        assert RateLimitingMiddleware.DEFAULT_LIMITS["health"] == "100/minute"
        assert RateLimitingMiddleware.DEFAULT_LIMITS["ai"] == "50/hour"
        assert RateLimitingMiddleware.DEFAULT_LIMITS["read"] == "200/minute"
        assert RateLimitingMiddleware.DEFAULT_LIMITS["write"] == "60/minute"

    def test_setup_rate_limiting(self):
        """Test rate limiting setup function."""
        app = MagicMock()
        app.state = MagicMock()
        app.add_exception_handler = MagicMock()

        limiter = setup_rate_limiting(app)

        assert app.state.limiter == limiter
        assert app.add_exception_handler.called
        assert app.add_exception_handler.call_args[0][0] == RateLimitExceeded
