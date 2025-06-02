"""Comprehensive test suite for Firebase Authentication Middleware.

Tests cover all aspects of Firebase authentication middleware functionality:
- Token extraction and validation
- User context creation and role-based access control
- Error handling scenarios
- Performance with caching
- Resource cleanup
- Integration with FastAPI
"""

import asyncio
from datetime import UTC, datetime, timedelta
import json
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import pytest
from starlette.responses import JSONResponse

from clarity.auth.firebase_auth import (
    FirebaseAuthMiddleware,
    FirebaseAuthProvider,
)
from clarity.auth.models import AuthError, Permission, UserContext, UserRole
from clarity.core.config import MiddlewareConfig

# Import Firebase types only for type checking to avoid stub warnings
if TYPE_CHECKING:
    from firebase_admin import auth as firebase_auth_module


class TestFirebaseAuthProvider:
    """Test suite for Firebase authentication provider."""

    @pytest.fixture
    def middleware_config(self) -> MiddlewareConfig:
        """Create test middleware configuration."""
        return MiddlewareConfig(
            enabled=True,
            cache_enabled=True,
            cache_ttl_seconds=300,
            cache_max_size=100,
            graceful_degradation=True,
            fallback_to_mock=True,
            initialization_timeout_seconds=5,
        )

    @pytest.fixture
    def auth_provider(
        self, middleware_config: MiddlewareConfig
    ) -> FirebaseAuthProvider:
        """Create Firebase auth provider with test configuration."""
        return FirebaseAuthProvider(
            credentials_path="test/path",
            project_id="test-project",
            middleware_config=middleware_config,
        )

    @pytest.fixture
    def mock_firebase_user_record(self) -> Mock:
        """Create mock Firebase user record."""
        user_record = Mock()
        user_record.uid = "test_user_123"
        user_record.email = "test@example.com"
        user_record.display_name = "Test User"
        user_record.email_verified = True
        user_record.disabled = False

        # Mock user metadata
        metadata = Mock()
        metadata.creation_timestamp = int(
            time.time() * 1000
        )  # Firebase uses milliseconds
        metadata.last_sign_in_timestamp = int(time.time() * 1000)
        user_record.user_metadata = metadata

        return user_record

    @pytest.fixture
    def mock_decoded_token(self) -> dict[str, Any]:
        """Create mock decoded Firebase token."""
        return {
            "uid": "test_user_123",
            "email": "test@example.com",
            "email_verified": True,
            "custom_claims": {"role": "patient"},
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        }

    @pytest.mark.asyncio
    async def test_verify_token_success(
        self,
        auth_provider: FirebaseAuthProvider,
        mock_firebase_user_record: Mock,
        mock_decoded_token: dict[str, Any],
    ) -> None:
        """Test successful token verification."""
        with (
            patch("clarity.auth.firebase_auth.auth.verify_id_token") as mock_verify,
            patch("clarity.auth.firebase_auth.auth.get_user") as mock_get_user,
            patch.object(auth_provider, "_ensure_initialized", new_callable=AsyncMock),
        ):
            mock_verify.return_value = mock_decoded_token
            mock_get_user.return_value = mock_firebase_user_record

            result = await auth_provider.verify_token("valid_token")

            assert result is not None
            assert result["user_id"] == "test_user_123"
            assert result["email"] == "test@example.com"
            assert result["name"] == "Test User"
            assert result["verified"] is True
            assert result["roles"] == ["patient"]
            assert "created_at" in result
            assert "custom_claims" in result

    @pytest.mark.asyncio
    async def test_verify_token_expired(
        self, auth_provider: FirebaseAuthProvider
    ) -> None:
        """Test token verification with expired token."""
        # Import Firebase auth locally to avoid stub issues
        with (
            patch("firebase_admin.auth.verify_id_token") as mock_verify,
            patch.object(auth_provider, "_ensure_initialized", new_callable=AsyncMock),
        ):
            # Create mock exception directly to avoid import issues
            mock_verify.side_effect = Exception(
                "Token expired"
            )  # Simplified for testing

            result = await auth_provider.verify_token("expired_token")

            assert result is None

    @pytest.mark.asyncio
    async def test_verify_token_revoked(
        self, auth_provider: FirebaseAuthProvider
    ) -> None:
        """Test token verification with revoked token."""
        with (
            patch("firebase_admin.auth.verify_id_token") as mock_verify,
            patch.object(auth_provider, "_ensure_initialized", new_callable=AsyncMock),
        ):
            # Create mock exception directly to avoid import issues
            mock_verify.side_effect = Exception(
                "Token revoked"
            )  # Simplified for testing

            result = await auth_provider.verify_token("revoked_token")

            assert result is None

    @pytest.mark.asyncio
    async def test_verify_token_invalid(
        self, auth_provider: FirebaseAuthProvider
    ) -> None:
        """Test token verification with invalid token."""
        with (
            patch("firebase_admin.auth.verify_id_token") as mock_verify,
            patch.object(auth_provider, "_ensure_initialized", new_callable=AsyncMock),
        ):
            # Create mock exception directly to avoid import issues
            mock_verify.side_effect = Exception(
                "Invalid token"
            )  # Simplified for testing

            result = await auth_provider.verify_token("invalid_token")

            assert result is None

    @pytest.mark.asyncio
    async def test_token_caching_enabled(
        self,
        auth_provider: FirebaseAuthProvider,
        mock_firebase_user_record: Mock,
        mock_decoded_token: dict[str, Any],
    ) -> None:
        """Test token caching when enabled."""
        with (
            patch("clarity.auth.firebase_auth.auth.verify_id_token") as mock_verify,
            patch("clarity.auth.firebase_auth.auth.get_user") as mock_get_user,
            patch.object(auth_provider, "_ensure_initialized", new_callable=AsyncMock),
        ):
            mock_verify.return_value = mock_decoded_token
            mock_get_user.return_value = mock_firebase_user_record

            # First call should hit Firebase
            result1 = await auth_provider.verify_token("cached_token")
            assert result1 is not None
            assert mock_verify.call_count == 1

            # Second call should use cache
            result2 = await auth_provider.verify_token("cached_token")
            assert result2 is not None
            assert result1 == result2
            assert mock_verify.call_count == 1  # Should not call Firebase again

    @pytest.mark.asyncio
    async def test_token_caching_disabled(self) -> None:
        """Test behavior when token caching is disabled."""
        config = MiddlewareConfig(cache_enabled=False)
        auth_provider = FirebaseAuthProvider(
            project_id="test-project",
            middleware_config=config,
        )

        mock_user_record = Mock()
        mock_user_record.uid = "test_user"
        mock_user_record.email = "test@example.com"
        mock_user_record.display_name = "Test"
        mock_user_record.email_verified = True
        mock_user_record.disabled = False
        mock_user_record.user_metadata = Mock()
        mock_user_record.user_metadata.creation_timestamp = int(time.time() * 1000)
        mock_user_record.user_metadata.last_sign_in_timestamp = int(time.time() * 1000)

        with (
            patch("clarity.auth.firebase_auth.auth.verify_id_token") as mock_verify,
            patch("clarity.auth.firebase_auth.auth.get_user") as mock_get_user,
            patch.object(auth_provider, "_ensure_initialized", new_callable=AsyncMock),
        ):
            mock_verify.return_value = {"uid": "test_user", "custom_claims": {}}
            mock_get_user.return_value = mock_user_record

            # Both calls should hit Firebase
            await auth_provider.verify_token("no_cache_token")
            await auth_provider.verify_token("no_cache_token")

            assert mock_verify.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_cleanup_expired_entries(
        self,
        auth_provider: FirebaseAuthProvider,
        mock_firebase_user_record: Mock,
        mock_decoded_token: dict[str, Any],
    ) -> None:
        """Test automatic cleanup of expired cache entries."""
        # Set short TTL for testing
        auth_provider._cache_ttl = 0.1  # 100ms

        with (
            patch("clarity.auth.firebase_auth.auth.verify_id_token") as mock_verify,
            patch("clarity.auth.firebase_auth.auth.get_user") as mock_get_user,
            patch.object(auth_provider, "_ensure_initialized", new_callable=AsyncMock),
        ):
            mock_verify.return_value = mock_decoded_token
            mock_get_user.return_value = mock_firebase_user_record

            # Cache a token
            await auth_provider.verify_token("expire_test_token")
            assert len(auth_provider._token_cache) == 1

            # Wait for expiration
            await asyncio.sleep(0.2)

            # Next verification should clean up expired entries
            await auth_provider.verify_token("new_token")

            # Should call Firebase twice due to expiration
            assert mock_verify.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_size_limit(
        self,
        auth_provider: FirebaseAuthProvider,
        mock_firebase_user_record: Mock,
        mock_decoded_token: dict[str, Any],
    ) -> None:
        """Test cache size limit enforcement."""
        # Set small cache size
        auth_provider._cache_max_size = 2

        with (
            patch("clarity.auth.firebase_auth.auth.verify_id_token") as mock_verify,
            patch("clarity.auth.firebase_auth.auth.get_user") as mock_get_user,
            patch.object(auth_provider, "_ensure_initialized", new_callable=AsyncMock),
        ):
            mock_verify.return_value = mock_decoded_token
            mock_get_user.return_value = mock_firebase_user_record

            # Fill cache beyond limit
            await auth_provider.verify_token("token1")
            await auth_provider.verify_token("token2")
            await auth_provider.verify_token("token3")  # Should evict oldest

            # Cache should not exceed max size
            assert len(auth_provider._token_cache) <= 2

    @pytest.mark.asyncio
    async def test_get_user_info_success(
        self,
        auth_provider: FirebaseAuthProvider,
        mock_firebase_user_record: Mock,
    ) -> None:
        """Test successful user info retrieval."""
        with (
            patch("clarity.auth.firebase_auth.auth.get_user") as mock_get_user,
            patch.object(auth_provider, "_ensure_initialized", new_callable=AsyncMock),
        ):
            mock_get_user.return_value = mock_firebase_user_record

            result = await auth_provider.get_user_info("test_user_123")

            assert result is not None
            assert result["user_id"] == "test_user_123"
            assert result["email"] == "test@example.com"

    @pytest.mark.asyncio
    async def test_get_user_info_not_found(
        self, auth_provider: FirebaseAuthProvider
    ) -> None:
        """Test user info retrieval for non-existent user."""
        from firebase_admin import auth

        with (
            patch("clarity.auth.firebase_auth.auth.get_user") as mock_get_user,
            patch.object(auth_provider, "_ensure_initialized", new_callable=AsyncMock),
        ):
            mock_get_user.side_effect = auth.UserNotFoundError("User not found")

            result = await auth_provider.get_user_info("nonexistent_user")

            assert result is None

    @pytest.mark.asyncio
    async def test_cleanup_resources(self, auth_provider: FirebaseAuthProvider) -> None:
        """Test proper cleanup of resources."""
        # Add some items to cache
        auth_provider._token_cache = {"token1": {"data": "test"}}

        await auth_provider.cleanup()

        assert len(auth_provider._token_cache) == 0


class TestFirebaseAuthMiddleware:
    """Test suite for Firebase authentication middleware."""

    @pytest.fixture
    def mock_auth_provider(self) -> Mock:
        """Create mock authentication provider."""
        provider = AsyncMock()
        provider.verify_token = AsyncMock()
        return provider

    @pytest.fixture
    def middleware(self, mock_auth_provider: Mock) -> FirebaseAuthMiddleware:
        """Create Firebase auth middleware with mock provider."""
        app = FastAPI()
        return FirebaseAuthMiddleware(
            app=app,
            auth_provider=mock_auth_provider,
            exempt_paths=["/health", "/docs", "/openapi.json"],
        )

    @pytest.fixture
    def mock_request(self) -> Mock:
        """Create mock FastAPI request."""
        request = Mock(spec=Request)
        request.url = Mock()
        request.url.path = "/api/protected"
        request.headers = {}
        request.state = Mock()
        return request

    @pytest.fixture
    def sample_user_info(self) -> dict[str, Any]:
        """Create sample user info for testing."""
        return {
            "user_id": "test_user_123",
            "email": "test@example.com",
            "name": "Test User",
            "verified": True,
            "roles": ["patient"],
            "custom_claims": {},
            "created_at": datetime.now(UTC).isoformat(),
            "last_login": None,
        }

    def test_extract_token_success(self, middleware: FirebaseAuthMiddleware) -> None:
        """Test successful token extraction from Authorization header."""
        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token_here"}

        token = middleware._extract_token(request)

        assert token == "valid_token_here"

    def test_extract_token_missing_header(
        self, middleware: FirebaseAuthMiddleware
    ) -> None:
        """Test token extraction with missing Authorization header."""
        request = Mock()
        request.headers = {}

        with pytest.raises(AuthError) as exc_info:
            middleware._extract_token(request)

        assert exc_info.value.status_code == 401
        assert exc_info.value.error_code == "missing_token"

    def test_extract_token_invalid_format(
        self, middleware: FirebaseAuthMiddleware
    ) -> None:
        """Test token extraction with invalid Authorization header format."""
        request = Mock()
        request.headers = {"Authorization": "Basic invalid_format"}

        with pytest.raises(AuthError) as exc_info:
            middleware._extract_token(request)

        assert exc_info.value.status_code == 401
        assert exc_info.value.error_code == "invalid_token_format"

    def test_extract_token_empty_token(
        self, middleware: FirebaseAuthMiddleware
    ) -> None:
        """Test token extraction with empty token."""
        request = Mock()
        request.headers = {"Authorization": "Bearer "}

        with pytest.raises(AuthError) as exc_info:
            middleware._extract_token(request)

        assert exc_info.value.status_code == 401
        assert exc_info.value.error_code == "empty_token"

    def test_is_exempt_path(self, middleware: FirebaseAuthMiddleware) -> None:
        """Test exempt path checking."""
        assert middleware._is_exempt_path("/health") is True
        assert middleware._is_exempt_path("/docs") is True
        assert middleware._is_exempt_path("/openapi.json") is True
        assert middleware._is_exempt_path("/api/protected") is False
        assert (
            middleware._is_exempt_path("/health/detailed") is True
        )  # Starts with /health

    def test_create_user_context_patient(
        self, middleware: FirebaseAuthMiddleware
    ) -> None:
        """Test user context creation for patient role."""
        user_info = {
            "user_id": "patient_123",
            "email": "patient@example.com",
            "name": "Patient User",
            "verified": True,
            "roles": ["patient"],
            "custom_claims": {},
            "created_at": datetime.now(UTC).isoformat(),
            "last_login": None,
        }

        context = middleware._create_user_context(user_info)

        assert context.user_id == "patient_123"
        assert context.email == "patient@example.com"
        assert context.role == UserRole.PATIENT
        assert Permission.READ_OWN_DATA in context.permissions
        assert Permission.WRITE_OWN_DATA in context.permissions
        assert Permission.READ_PATIENT_DATA not in context.permissions

    def test_create_user_context_clinician(
        self, middleware: FirebaseAuthMiddleware
    ) -> None:
        """Test user context creation for clinician role."""
        user_info = {
            "user_id": "clinician_123",
            "email": "clinician@example.com",
            "name": "Dr. Clinician",
            "verified": True,
            "roles": ["clinician"],
            "custom_claims": {},
            "created_at": datetime.now(UTC).isoformat(),
            "last_login": None,
        }

        context = middleware._create_user_context(user_info)

        assert context.role == UserRole.CLINICIAN
        assert Permission.READ_OWN_DATA in context.permissions
        assert Permission.WRITE_OWN_DATA in context.permissions
        assert Permission.READ_PATIENT_DATA in context.permissions
        assert Permission.WRITE_PATIENT_DATA in context.permissions
        assert Permission.SYSTEM_ADMIN not in context.permissions

    def test_create_user_context_admin(
        self, middleware: FirebaseAuthMiddleware
    ) -> None:
        """Test user context creation for admin role."""
        user_info = {
            "user_id": "admin_123",
            "email": "admin@example.com",
            "name": "Admin User",
            "verified": True,
            "roles": ["admin"],
            "custom_claims": {},
            "created_at": datetime.now(UTC).isoformat(),
            "last_login": None,
        }

        context = middleware._create_user_context(user_info)

        assert context.role == UserRole.ADMIN
        assert Permission.SYSTEM_ADMIN in context.permissions
        assert Permission.MANAGE_USERS in context.permissions
        assert len(context.permissions) == 7  # All permissions

    def test_create_user_context_invalid_role(
        self, middleware: FirebaseAuthMiddleware
    ) -> None:
        """Test user context creation with invalid role defaults to patient."""
        user_info = {
            "user_id": "user_123",
            "email": "user@example.com",
            "name": "User",
            "verified": True,
            "roles": ["invalid_role"],
            "custom_claims": {},
            "created_at": datetime.now(UTC).isoformat(),
            "last_login": None,
        }

        context = middleware._create_user_context(user_info)

        assert context.role == UserRole.PATIENT  # Should default to patient

    @pytest.mark.asyncio
    async def test_authenticate_request_success(
        self,
        middleware: FirebaseAuthMiddleware,
        mock_auth_provider: Mock,
        mock_request: Mock,
        sample_user_info: dict[str, Any],
    ) -> None:
        """Test successful request authentication."""
        mock_request.headers = {"Authorization": "Bearer valid_token"}
        mock_auth_provider.verify_token.return_value = sample_user_info
        middleware.auth_provider = mock_auth_provider

        user_context = await middleware._authenticate_request(mock_request)

        assert isinstance(user_context, UserContext)
        assert user_context.user_id == "test_user_123"
        assert user_context.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_authenticate_request_invalid_token(
        self,
        middleware: FirebaseAuthMiddleware,
        mock_auth_provider: Mock,
        mock_request: Mock,
    ) -> None:
        """Test request authentication with invalid token."""
        mock_request.headers = {"Authorization": "Bearer invalid_token"}
        mock_auth_provider.verify_token.return_value = None
        middleware.auth_provider = mock_auth_provider

        with pytest.raises(AuthError) as exc_info:
            await middleware._authenticate_request(mock_request)

        assert exc_info.value.status_code == 401
        assert exc_info.value.error_code == "invalid_token"

    @pytest.mark.asyncio
    async def test_dispatch_exempt_path(
        self,
        middleware: FirebaseAuthMiddleware,
        mock_auth_provider: Mock,
    ) -> None:
        """Test middleware dispatch for exempt paths."""
        request = Mock()
        request.url.path = "/health"

        async def call_next(req: Request) -> JSONResponse:
            return JSONResponse({"status": "ok"})

        response = await middleware.dispatch(request, call_next)

        assert response.status_code == 200
        # Should not call auth provider for exempt paths
        mock_auth_provider.verify_token.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_protected_path_success(
        self,
        middleware: FirebaseAuthMiddleware,
        mock_auth_provider: Mock,
        sample_user_info: dict[str, Any],
    ) -> None:
        """Test middleware dispatch for protected paths with valid auth."""
        request = Mock()
        request.url.path = "/api/protected"
        request.headers = {"Authorization": "Bearer valid_token"}
        request.state = Mock()

        mock_auth_provider.verify_token.return_value = sample_user_info
        middleware.auth_provider = mock_auth_provider

        async def call_next(req: Request) -> JSONResponse:
            # Verify user context was attached
            assert hasattr(req.state, "user")
            assert isinstance(req.state.user, UserContext)
            return JSONResponse({"status": "authorized"})

        response = await middleware.dispatch(request, call_next)

        assert response.status_code == 200
        mock_auth_provider.verify_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_protected_path_auth_error(
        self,
        middleware: FirebaseAuthMiddleware,
        mock_auth_provider: Mock,
    ) -> None:
        """Test middleware dispatch for protected paths with auth error."""
        request = Mock()
        request.url.path = "/api/protected"
        request.headers = {}  # Missing Authorization header

        async def call_next(req: Request) -> JSONResponse:
            return JSONResponse({"status": "should_not_reach"})

        response = await middleware.dispatch(request, call_next)

        assert response.status_code == 401

        # Parse response content
        content = json.loads(response.body.decode())
        assert content["error"] == "missing_token"
        assert "timestamp" in content


class TestIntegrationFirebaseAuth:
    """Integration tests for Firebase authentication with FastAPI."""

    @pytest.fixture
    def app_with_auth(self) -> FastAPI:
        """Create FastAPI app with Firebase auth middleware."""
        app = FastAPI()

        # Create mock auth provider
        mock_provider = AsyncMock()
        mock_provider.verify_token = AsyncMock()

        # Add middleware
        middleware = FirebaseAuthMiddleware(
            app=app,
            auth_provider=mock_provider,
            exempt_paths=["/health", "/public"],
        )

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "healthy"}

        @app.get("/public")
        async def public_endpoint() -> dict[str, str]:
            return {"message": "public access"}

        @app.get("/protected")
        async def protected_endpoint(request: Request) -> dict[str, str]:
            user = request.state.user
            return {"message": f"Hello {user.email}"}

        # Store reference to mock provider for test access
        app.state.mock_auth_provider = mock_provider

        return app

    def test_public_endpoint_access(self, app_with_auth: FastAPI) -> None:
        """Test access to public endpoints without authentication."""
        client = TestClient(app_with_auth)

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

        response = client.get("/public")
        assert response.status_code == 200
        assert response.json() == {"message": "public access"}

    def test_protected_endpoint_without_auth(self, app_with_auth: FastAPI) -> None:
        """Test access to protected endpoint without authentication."""
        client = TestClient(app_with_auth)

        response = client.get("/protected")
        assert response.status_code == 401

    def test_protected_endpoint_with_valid_auth(self, app_with_auth: FastAPI) -> None:
        """Test access to protected endpoint with valid authentication."""
        # Mock successful authentication
        mock_user_info = {
            "user_id": "test_user",
            "email": "test@example.com",
            "name": "Test User",
            "verified": True,
            "roles": ["patient"],
            "custom_claims": {},
            "created_at": datetime.now(UTC).isoformat(),
            "last_login": None,
        }

        app_with_auth.state.mock_auth_provider.verify_token.return_value = (
            mock_user_info
        )

        client = TestClient(app_with_auth)
        response = client.get(
            "/protected", headers={"Authorization": "Bearer valid_token"}
        )

        assert response.status_code == 200
        assert response.json() == {"message": "Hello test@example.com"}

    def test_protected_endpoint_with_invalid_auth(self, app_with_auth: FastAPI) -> None:
        """Test access to protected endpoint with invalid authentication."""
        # Mock failed authentication
        app_with_auth.state.mock_auth_provider.verify_token.return_value = None

        client = TestClient(app_with_auth)
        response = client.get(
            "/protected", headers={"Authorization": "Bearer invalid_token"}
        )

        assert response.status_code == 401


class TestPerformanceFirebaseAuth:
    """Performance tests for Firebase authentication."""

    @pytest.mark.asyncio
    async def test_token_verification_performance_with_cache(self) -> None:
        """Test token verification performance with caching enabled."""
        config = MiddlewareConfig(cache_enabled=True, cache_ttl_seconds=300)
        auth_provider = FirebaseAuthProvider(
            project_id="test-project",
            middleware_config=config,
        )

        mock_user_record = Mock()
        mock_user_record.uid = "perf_user"
        mock_user_record.email = "perf@example.com"
        mock_user_record.display_name = "Performance User"
        mock_user_record.email_verified = True
        mock_user_record.disabled = False
        mock_user_record.user_metadata = Mock()
        mock_user_record.user_metadata.creation_timestamp = int(time.time() * 1000)
        mock_user_record.user_metadata.last_sign_in_timestamp = int(time.time() * 1000)

        with (
            patch("clarity.auth.firebase_auth.auth.verify_id_token") as mock_verify,
            patch("clarity.auth.firebase_auth.auth.get_user") as mock_get_user,
            patch.object(auth_provider, "_ensure_initialized", new_callable=AsyncMock),
        ):
            mock_verify.return_value = {"uid": "perf_user", "custom_claims": {}}
            mock_get_user.return_value = mock_user_record

            # Measure first call (should hit Firebase)
            start_time = time.time()
            await auth_provider.verify_token("performance_token")
            first_call_time = time.time() - start_time

            # Measure second call (should use cache)
            start_time = time.time()
            await auth_provider.verify_token("performance_token")
            second_call_time = time.time() - start_time

            # Cached call should be significantly faster
            assert second_call_time < first_call_time
            assert mock_verify.call_count == 1  # Only called once due to caching

    @pytest.mark.asyncio
    async def test_concurrent_token_verification(self) -> None:
        """Test concurrent token verification requests."""
        config = MiddlewareConfig(cache_enabled=True)
        auth_provider = FirebaseAuthProvider(
            project_id="test-project",
            middleware_config=config,
        )

        mock_user_record = Mock()
        mock_user_record.uid = "concurrent_user"
        mock_user_record.email = "concurrent@example.com"
        mock_user_record.display_name = "Concurrent User"
        mock_user_record.email_verified = True
        mock_user_record.disabled = False
        mock_user_record.user_metadata = Mock()
        mock_user_record.user_metadata.creation_timestamp = int(time.time() * 1000)
        mock_user_record.user_metadata.last_sign_in_timestamp = int(time.time() * 1000)

        with (
            patch("clarity.auth.firebase_auth.auth.verify_id_token") as mock_verify,
            patch("clarity.auth.firebase_auth.auth.get_user") as mock_get_user,
            patch.object(auth_provider, "_ensure_initialized", new_callable=AsyncMock),
        ):
            mock_verify.return_value = {"uid": "concurrent_user", "custom_claims": {}}
            mock_get_user.return_value = mock_user_record

            # Run multiple concurrent token verifications
            tasks = [
                auth_provider.verify_token(f"concurrent_token_{i}") for i in range(10)
            ]

            results = await asyncio.gather(*tasks)

            # All should succeed
            assert all(result is not None for result in results)
            # Should have made requests for each unique token
            assert mock_verify.call_count == 10


# Performance benchmark helper
async def benchmark_auth_performance() -> dict[str, float]:
    """Benchmark authentication performance for different scenarios."""
    results = {}

    # Test with caching enabled
    cached_config = MiddlewareConfig(cache_enabled=True)
    cached_provider = FirebaseAuthProvider(
        project_id="benchmark",
        middleware_config=cached_config,
    )

    # Test without caching
    no_cache_config = MiddlewareConfig(cache_enabled=False)
    no_cache_provider = FirebaseAuthProvider(
        project_id="benchmark",
        middleware_config=no_cache_config,
    )

    return results


if __name__ == "__main__":
    # Run performance benchmark
    import asyncio

    async def run_benchmark() -> None:
        results = await benchmark_auth_performance()
        print("Firebase Auth Performance Benchmark Results:")
        for test_name, duration in results.items():
            print(f"  {test_name}: {duration:.4f}s")

    # asyncio.run(run_benchmark())
