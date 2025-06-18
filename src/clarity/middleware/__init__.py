"""Middleware package for CLARITY backend.

This package contains various middleware components for cross-cutting concerns
such as authentication, logging, and error handling.
"""

from clarity.middleware.auth_middleware import CognitoAuthMiddleware
from clarity.middleware.rate_limiting import (
    RateLimitingMiddleware,
    get_ip_only,
    get_user_id_or_ip,
    rate_limit,
    setup_rate_limiting,
)
from clarity.middleware.request_logger import RequestLoggingMiddleware
from clarity.middleware.security_headers import (
    SecurityHeadersMiddleware,
    setup_security_headers,
)

__all__ = [
    "CognitoAuthMiddleware",
    "RateLimitingMiddleware",
    "RequestLoggingMiddleware",
    "SecurityHeadersMiddleware",
    "get_ip_only",
    "get_user_id_or_ip",
    "rate_limit",
    "setup_rate_limiting",
    "setup_security_headers",
]
