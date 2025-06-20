"""Test security headers middleware functionality."""

from __future__ import annotations

from typing import Never

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import Response

from clarity.middleware.security_headers import (
    SecurityHeadersMiddleware,
    setup_security_headers,
)


def create_test_app(
    *,
    enable_hsts: bool = True,
    enable_csp: bool = True,
    cache_control: str = "no-store, private",
) -> FastAPI:
    """Create a test FastAPI app with security headers middleware."""
    app = FastAPI()

    # Add the security headers middleware
    app.add_middleware(
        SecurityHeadersMiddleware,
        enable_hsts=enable_hsts,
        enable_csp=enable_csp,
        cache_control=cache_control,
    )

    @app.get("/test")
    async def test_endpoint() -> dict[str, str]:
        return {"message": "test"}

    @app.get("/health")
    async def health_endpoint() -> dict[str, str]:
        return {"status": "healthy"}

    return app


class TestSecurityHeadersMiddleware:
    """Test cases for security headers middleware."""

    def test_all_security_headers_present(self):
        """Test that all security headers are present in responses."""
        app = create_test_app()
        client = TestClient(app)

        response = client.get("/test")

        # Check all expected headers
        assert "Strict-Transport-Security" in response.headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "Content-Security-Policy" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert "Referrer-Policy" in response.headers
        assert "Cache-Control" in response.headers
        assert "Permissions-Policy" in response.headers

    def test_hsts_header_values(self):
        """Test HSTS header has correct values."""
        app = create_test_app()
        client = TestClient(app)

        response = client.get("/test")

        hsts = response.headers["Strict-Transport-Security"]
        assert "max-age=31536000" in hsts  # 1 year
        assert "includeSubDomains" in hsts

    def test_security_header_values(self):
        """Test that security headers have correct values."""
        app = create_test_app()
        client = TestClient(app)

        response = client.get("/test")

        # Check specific header values
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert (
            response.headers["Content-Security-Policy"]
            == 'default-src "none"; frame-ancestors "none";'
        )
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert response.headers["Cache-Control"] == "no-store, private"

        # Check permissions policy contains expected values
        permissions = response.headers["Permissions-Policy"]
        assert "camera=()" in permissions
        assert "microphone=()" in permissions
        assert "geolocation=()" in permissions
        assert "payment=()" in permissions
        assert "usb=()" in permissions

    def test_hsts_disabled(self):
        """Test that HSTS header is not added when disabled."""
        app = create_test_app(enable_hsts=False)
        client = TestClient(app)

        response = client.get("/test")

        # HSTS should not be present
        assert "Strict-Transport-Security" not in response.headers

        # Other headers should still be present
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers

    def test_csp_disabled(self):
        """Test that CSP header is not added when disabled."""
        app = create_test_app(enable_csp=False)
        client = TestClient(app)

        response = client.get("/test")

        # CSP should not be present
        assert "Content-Security-Policy" not in response.headers

        # Other headers should still be present
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers

    def test_custom_cache_control(self):
        """Test custom cache control header."""
        custom_cache = "max-age=3600, private"
        app = create_test_app(cache_control=custom_cache)
        client = TestClient(app)

        response = client.get("/test")

        assert response.headers["Cache-Control"] == custom_cache

    def test_headers_on_different_endpoints(self):
        """Test that headers are added to all endpoints."""
        app = create_test_app()
        client = TestClient(app)

        # Test multiple endpoints
        for endpoint in ["/test", "/health"]:
            response = client.get(endpoint)

            # Check key headers are present
            assert "X-Content-Type-Options" in response.headers
            assert "X-Frame-Options" in response.headers
            assert response.headers["X-Content-Type-Options"] == "nosniff"

    def test_headers_on_different_methods(self):
        """Test that headers are added to all HTTP methods."""
        app = create_test_app()

        @app.post("/test")
        async def post_endpoint() -> dict[str, str]:  # noqa: RUF029 - Test handler
            return {"message": "posted"}

        client = TestClient(app)

        # Test different HTTP methods
        get_response = client.get("/test")
        post_response = client.post("/test")

        # Both should have security headers
        for response in [get_response, post_response]:
            assert "X-Content-Type-Options" in response.headers
            assert "X-Frame-Options" in response.headers

    def test_setup_security_headers_helper(self):
        """Test the setup_security_headers helper function."""
        app = FastAPI()

        # Use the helper function
        middleware = setup_security_headers(
            app,
            enable_hsts=True,
            enable_csp=True,
            cache_control="no-cache",
        )

        # Check middleware properties
        assert isinstance(middleware, SecurityHeadersMiddleware)
        assert middleware.enable_hsts is True
        assert middleware.enable_csp is True
        assert middleware.cache_control == "no-cache"

    def test_custom_hsts_configuration(self):
        """Test custom HSTS configuration."""
        app = FastAPI()

        # Custom HSTS settings
        app.add_middleware(
            SecurityHeadersMiddleware,
            enable_hsts=True,
            hsts_max_age=86400,  # 1 day
            hsts_include_subdomains=False,
        )

        @app.get("/test")
        async def test_endpoint() -> dict[str, str]:
            return {"message": "test"}

        client = TestClient(app)
        response = client.get("/test")

        hsts = response.headers["Strict-Transport-Security"]
        assert "max-age=86400" in hsts
        assert "includeSubDomains" not in hsts

    def test_custom_csp_policy(self):
        """Test custom CSP policy."""
        app = FastAPI()
        custom_csp = "default-src 'self'; script-src 'self' 'unsafe-inline';"

        app.add_middleware(
            SecurityHeadersMiddleware,
            enable_csp=True,
            csp_policy=custom_csp,
        )

        @app.get("/test")
        async def test_endpoint() -> dict[str, str]:
            return {"message": "test"}

        client = TestClient(app)
        response = client.get("/test")

        assert response.headers["Content-Security-Policy"] == custom_csp

    def test_middleware_preserves_response_content(self):
        """Test that middleware doesn't modify response content."""
        app = create_test_app()
        client = TestClient(app)

        response = client.get("/test")

        # Check response content is unchanged
        assert response.status_code == 200
        assert response.json() == {"message": "test"}

    def test_middleware_handles_errors(self):
        """Test that middleware adds headers even on error responses."""
        app = FastAPI()

        # Add the security headers middleware
        app.add_middleware(SecurityHeadersMiddleware)

        # Add exception handler to return proper 500 response
        @app.exception_handler(ValueError)
        def value_error_handler(_request: Request, exc: ValueError) -> Response:
            return Response(content=str(exc), status_code=500)

        @app.get("/error", response_model=None)
        async def error_endpoint() -> Never:
            msg = "Test error"
            raise ValueError(msg)

        client = TestClient(app)

        # Even error responses should have security headers
        response = client.get("/error")
        assert response.status_code == 500
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
