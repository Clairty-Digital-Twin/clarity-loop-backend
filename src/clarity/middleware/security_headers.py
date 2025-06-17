"""Security headers middleware for CLARITY backend.

This middleware adds security headers to all HTTP responses to enhance security posture.
Implements OWASP recommended security headers for API protection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses.
    
    Implements industry-standard security headers to protect against common attacks:
    - XSS attacks
    - Clickjacking
    - MIME type sniffing
    - Information disclosure
    - Protocol downgrade attacks
    """

    def __init__(
        self,
        app: ASGIApp,
        enable_hsts: bool = True,
        hsts_max_age: int = 31536000,  # 1 year
        hsts_include_subdomains: bool = True,
        enable_csp: bool = True,
        csp_policy: str | None = None,
        cache_control: str = "no-store, private",
    ) -> None:
        """Initialize security headers middleware.
        
        Args:
            app: The ASGI application
            enable_hsts: Whether to enable HTTP Strict Transport Security
            hsts_max_age: Max age for HSTS in seconds (default: 1 year)
            hsts_include_subdomains: Whether to include subdomains in HSTS
            enable_csp: Whether to enable Content Security Policy
            csp_policy: Custom CSP policy (default: API-specific policy)
            cache_control: Cache control header value
        """
        super().__init__(app)
        self.enable_hsts = enable_hsts
        self.hsts_max_age = hsts_max_age
        self.hsts_include_subdomains = hsts_include_subdomains
        self.enable_csp = enable_csp
        self.csp_policy = csp_policy or 'default-src "none"; frame-ancestors "none";'
        self.cache_control = cache_control
        
        # Log configuration
        logger.info(
            "SecurityHeadersMiddleware initialized - HSTS: %s, CSP: %s",
            self.enable_hsts,
            self.enable_csp
        )

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process the request and add security headers to the response.
        
        Args:
            request: The incoming request
            call_next: The next middleware or endpoint handler
            
        Returns:
            Response with security headers added
        """
        # Process the request
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response)
        
        return response

    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to the response.
        
        Args:
            response: The response object to add headers to
        """
        # HSTS - Enforce HTTPS (only if enabled)
        if self.enable_hsts:
            hsts_value = f"max-age={self.hsts_max_age}"
            if self.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"
            response.headers["Strict-Transport-Security"] = hsts_value

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # Content Security Policy (API-specific)
        if self.enable_csp:
            response.headers["Content-Security-Policy"] = self.csp_policy
        
        # XSS Protection (legacy but still useful)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Control referrer information
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Prevent caching of sensitive data
        response.headers["Cache-Control"] = self.cache_control
        
        # Permissions Policy - Deny access to browser features
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), "
            "payment=(), usb=(), magnetometer=(), "
            "accelerometer=(), gyroscope=()"
        )


# Convenience function for easy registration
def setup_security_headers(
    app: ASGIApp,
    enable_hsts: bool = True,
    enable_csp: bool = True,
    cache_control: str = "no-store, private",
) -> SecurityHeadersMiddleware:
    """Setup security headers middleware with sensible defaults.
    
    Args:
        app: The ASGI application
        enable_hsts: Whether to enable HSTS (default: True)
        enable_csp: Whether to enable CSP (default: True)
        cache_control: Cache control policy (default: no-store, private)
        
    Returns:
        Configured SecurityHeadersMiddleware instance
    """
    return SecurityHeadersMiddleware(
        app,
        enable_hsts=enable_hsts,
        enable_csp=enable_csp,
        cache_control=cache_control,
    ) 