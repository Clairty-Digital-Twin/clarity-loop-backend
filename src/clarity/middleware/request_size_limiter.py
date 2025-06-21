"""Request Size Limiting Middleware for DoS Protection.

Prevents denial-of-service attacks by enforcing request body size limits
across all endpoints. Configurable limits based on content type and environment.
"""

# removed - breaks FastAPI

import logging
from typing import TYPE_CHECKING

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RequestSizeLimiterMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request body size limits for DoS protection.

    Prevents attackers from overwhelming the server with massive request payloads.
    Configurable limits based on content type and environment settings.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB default
        max_json_size: int = 5 * 1024 * 1024,  # 5MB for JSON payloads
        max_upload_size: int = 50 * 1024 * 1024,  # 50MB for file uploads
        max_form_size: int = 1024 * 1024,  # 1MB for form data
    ) -> None:
        """Initialize request size limiter.

        Args:
            app: FastAPI application
            max_request_size: Default maximum request size in bytes
            max_json_size: Maximum size for JSON payloads in bytes
            max_upload_size: Maximum size for file uploads in bytes
            max_form_size: Maximum size for form data in bytes
        """
        super().__init__(app)
        self.max_request_size = max_request_size
        self.max_json_size = max_json_size
        self.max_upload_size = max_upload_size
        self.max_form_size = max_form_size

        # Log configuration for security audit
        logger.info(
            "🔒 Request Size Limiter: Default max size: %d MB",
            max_request_size // (1024 * 1024),
        )
        logger.info(
            "🔒 Request Size Limiter: JSON max size: %d MB",
            max_json_size // (1024 * 1024),
        )
        logger.info(
            "🔒 Request Size Limiter: Upload max size: %d MB",
            max_upload_size // (1024 * 1024),
        )
        logger.info(
            "🔒 Request Size Limiter: Form max size: %d KB", max_form_size // 1024
        )

    def _get_size_limit(self, request: Request) -> int:
        """Determine the appropriate size limit based on request content type.

        Args:
            request: Incoming HTTP request

        Returns:
            Maximum allowed size in bytes for this request type
        """
        content_type = request.headers.get("content-type", "").lower()

        # File upload endpoints - higher limit
        if "multipart/form-data" in content_type:
            return self.max_upload_size
        # JSON API endpoints - moderate limit
        if "application/json" in content_type:
            return self.max_json_size
        # Form data - lower limit
        if "application/x-www-form-urlencoded" in content_type:
            return self.max_form_size
        # Default limit for unknown content types
        return self.max_request_size

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Check request size before processing.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/endpoint in chain

        Returns:
            HTTP response (413 if payload too large, otherwise normal response)
        """
        # Debug logging to see if middleware is called
        logger.info("🔍 Request Size Limiter: %s %s", request.method, request.url.path)

        # Only check requests with bodies (POST, PUT, PATCH)
        if request.method in {"POST", "PUT", "PATCH"}:
            # Check Content-Length header first (fastest check)
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    size = int(content_length)
                    limit = self._get_size_limit(request)

                    if size > limit:
                        limit_mb = limit / (1024 * 1024)
                        size_mb = size / (1024 * 1024)

                        # Log security incident
                        logger.warning(
                            "🚨 Request size limit exceeded: %.2f MB > %.2f MB limit for %s %s",
                            size_mb,
                            limit_mb,
                            request.method,
                            request.url.path,
                        )

                        # Return 413 Payload Too Large with security headers
                        from clarity.middleware.security_headers import (  # noqa: PLC0415
                            SecurityHeadersMiddleware,
                        )

                        response = JSONResponse(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            content={
                                "error": "Request payload too large",
                                "max_size_mb": round(limit_mb, 2),
                                "received_size_mb": round(size_mb, 2),
                                "content_type": request.headers.get(
                                    "content-type", "unknown"
                                ),
                                "message": f"Request size {size_mb:.1f}MB exceeds {limit_mb:.1f}MB limit",
                            },
                            headers={"Retry-After": "3600"},  # Suggest retry in 1 hour
                        )

                        # Add security headers to the response
                        SecurityHeadersMiddleware.add_security_headers_to_response(
                            response
                        )

                        return response

                except ValueError:
                    # Invalid Content-Length header
                    logger.warning("Invalid Content-Length header: %s", content_length)

        # Process request normally if size is acceptable
        return await call_next(request)
