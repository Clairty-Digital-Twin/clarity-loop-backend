"""Middleware package for CLARITY backend.

This package contains various middleware components for cross-cutting concerns
such as authentication, logging, and error handling.
"""

from clarity.middleware.auth_middleware import CognitoAuthMiddleware
from clarity.middleware.request_logger import RequestLoggingMiddleware

__all__ = ["CognitoAuthMiddleware", "RequestLoggingMiddleware"]
