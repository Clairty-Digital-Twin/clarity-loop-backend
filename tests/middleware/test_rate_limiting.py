"""Test rate limiting middleware."""

from unittest.mock import Mock

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
        assert result == "user:user123"

    def test_get_user_id_or_ip_with_uid(self):
        """Test user ID extraction with uid field."""
        mock_request = Mock(spec=Request)
        mock_request.state.user = {"uid": "uid456"}

        result = get_user_id_or_ip(mock_request)
        assert result == "user:uid456"

    def test_get_user_id_or_ip_fallback_to_ip(self):
        """Test fallback to IP when no user."""
        mock_request = Mock(spec=Request)
        mock_request.state.user = None
        mock_request.client = Mock(host="192.168.1.1")

        result = get_user_id_or_ip(mock_request)
        assert result == "ip:192.168.1.1"

    def test_get_ip_only_with_client(self):
        """Test IP extraction with client."""
        mock_request = Mock(spec=Request)
        mock_request.client = Mock(host="10.0.0.1")

        result = get_ip_only(mock_request)
        assert result == "ip:10.0.0.1"

    def test_get_ip_only_no_client(self):
        """Test IP extraction without client."""
        mock_request = Mock(spec=Request)
        mock_request.client = None

        result = get_ip_only(mock_request)
        assert result == "ip:127.0.0.1"


class TestRateLimitingMiddleware:
    """Test RateLimitingMiddleware class."""

    def test_middleware_class_exists(self):
        """Test that RateLimitingMiddleware class exists."""
        assert RateLimitingMiddleware is not None

    def test_create_limiter(self):
        """Test create limiter factory method."""
        limiter = RateLimitingMiddleware.create_limiter(["100/hour"])
        assert limiter is not None

    def test_get_auth_limiter(self):
        """Test auth limiter factory method."""
        limiter = RateLimitingMiddleware.get_auth_limiter()
        assert limiter is not None

    def test_get_ai_limiter(self):
        """Test AI limiter factory method."""
        limiter = RateLimitingMiddleware.get_ai_limiter()
        assert limiter is not None
