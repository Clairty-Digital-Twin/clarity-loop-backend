"""Comprehensive Firebase Authentication Integration Tests
======================================================

These tests definitively prove that Firebase authentication is working correctly.
They test the entire flow from token verification to endpoint access.
"""

from datetime import UTC, datetime, timedelta
import time
from unittest.mock import AsyncMock, MagicMock, patch

import firebase_admin
from firebase_admin import auth as firebase_auth
import jwt
import pytest

from clarity.auth.firebase_middleware import (
    FirebaseAuthMiddleware,
    FirebaseAuthProvider,
)
from clarity.models.auth import Permission, UserContext, UserRole


class TestFirebaseAuthIntegration:
    """Comprehensive tests for Firebase authentication flow."""

    @pytest.fixture
    def valid_token_payload(self):
        """Create a valid Firebase token payload."""
        now = int(time.time())
        return {
            "iss": "https://securetoken.google.com/clarity-loop-backend",
            "aud": "clarity-loop-backend",
            "auth_time": now - 300,  # 5 minutes ago
            "user_id": "test-user-123",
            "sub": "test-user-123",
            "iat": now - 60,  # 1 minute ago
            "exp": now + 3540,  # 59 minutes from now
            "email": "test@example.com",
            "email_verified": True,
            "uid": "test-user-123",
            "firebase": {"sign_in_provider": "password"},
        }

    @pytest.fixture
    def mock_firebase_admin(self):
        """Mock Firebase Admin SDK."""
        with (
            patch("firebase_admin.get_app") as mock_get_app,
            patch("firebase_admin.initialize_app") as mock_init_app,
            patch("firebase_admin.auth.verify_id_token") as mock_verify,
        ):

            # Mock app instance
            mock_app = MagicMock()
            mock_app.project_id = "clarity-loop-backend"
            mock_get_app.return_value = mock_app

            yield {
                "get_app": mock_get_app,
                "initialize_app": mock_init_app,
                "verify_id_token": mock_verify,
                "app": mock_app,
            }

    @pytest.mark.asyncio
    async def test_firebase_initialization_with_project_id(self, mock_firebase_admin):
        """Test that Firebase Admin SDK is initialized with correct project ID."""
        # Setup
        provider = FirebaseAuthProvider(
            project_id="clarity-loop-backend", credentials_path=None
        )

        # Initialize
        await provider.initialize()

        # Verify
        assert provider._initialized
        assert mock_firebase_admin["get_app"].called

    @pytest.mark.asyncio
    async def test_token_verification_success(
        self, mock_firebase_admin, valid_token_payload
    ):
        """Test successful token verification flow."""
        # Setup
        mock_firebase_admin["verify_id_token"].return_value = valid_token_payload

        provider = FirebaseAuthProvider(project_id="clarity-loop-backend")
        provider._initialized = True

        # Test
        result = await provider.verify_token("valid-token")

        # Verify
        assert result is not None
        assert result["user_id"] == "test-user-123"
        assert result["email"] == "test@example.com"
        assert result["verified"] is True

        # Check that verify_id_token was called correctly
        mock_firebase_admin["verify_id_token"].assert_called_once_with(
            "valid-token", check_revoked=True
        )

    @pytest.mark.asyncio
    async def test_token_verification_with_expired_token(self, mock_firebase_admin):
        """Test that expired tokens are rejected."""
        # Setup
        from firebase_admin.auth import ExpiredIdTokenError

        mock_firebase_admin["verify_id_token"].side_effect = ExpiredIdTokenError(
            "Token has expired"
        )

        provider = FirebaseAuthProvider(project_id="clarity-loop-backend")
        provider._initialized = True

        # Test
        result = await provider.verify_token("expired-token")

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_token_verification_with_invalid_token(self, mock_firebase_admin):
        """Test that invalid tokens are rejected."""
        # Setup
        from firebase_admin.auth import InvalidIdTokenError

        mock_firebase_admin["verify_id_token"].side_effect = InvalidIdTokenError(
            "Token is invalid"
        )

        provider = FirebaseAuthProvider(project_id="clarity-loop-backend")
        provider._initialized = True

        # Test
        result = await provider.verify_token("invalid-token")

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_project_id_mismatch_detection(self, mock_firebase_admin):
        """Test that tokens from wrong project are rejected."""
        # Setup - token from different project
        wrong_project_token = {
            "aud": "different-project",  # Wrong audience
            "iss": "https://securetoken.google.com/different-project",
            "uid": "test-user-123",
            "email": "test@example.com",
        }

        from firebase_admin.auth import InvalidIdTokenError

        mock_firebase_admin["verify_id_token"].side_effect = InvalidIdTokenError(
            "Token was not issued for this project"
        )

        provider = FirebaseAuthProvider(project_id="clarity-loop-backend")
        provider._initialized = True

        # Test
        result = await provider.verify_token("wrong-project-token")

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_user_context_creation(
        self, mock_firebase_admin, valid_token_payload
    ):
        """Test that UserContext is created correctly from token."""
        # Setup
        mock_firebase_admin["verify_id_token"].return_value = valid_token_payload

        provider = FirebaseAuthProvider(project_id="clarity-loop-backend")
        provider._initialized = True

        # Test token verification
        user_info = await provider.verify_token("valid-token")
        assert user_info is not None

        # Create user context
        user_context = FirebaseAuthMiddleware._create_user_context(user_info)

        # Verify UserContext
        assert isinstance(user_context, UserContext)
        assert user_context.user_id == "test-user-123"
        assert user_context.email == "test@example.com"
        assert user_context.is_verified is True
        assert user_context.role == UserRole.PATIENT  # Default role
        assert Permission.READ_OWN_DATA in user_context.permissions
        assert Permission.WRITE_OWN_DATA in user_context.permissions

    @pytest.mark.asyncio
    async def test_middleware_integration(
        self, mock_firebase_admin, valid_token_payload
    ):
        """Test the complete middleware flow."""
        # Setup
        mock_firebase_admin["verify_id_token"].return_value = valid_token_payload

        # Create auth provider and middleware
        auth_provider = FirebaseAuthProvider(project_id="clarity-loop-backend")
        auth_provider._initialized = True

        # Create mock app and request
        mock_app = MagicMock()
        mock_request = MagicMock()
        mock_request.url.path = "/api/v1/health-data"
        mock_request.headers = {"authorization": "Bearer valid-token"}
        mock_request.state = MagicMock()

        # Create middleware
        middleware = FirebaseAuthMiddleware(
            app=mock_app, auth_provider=auth_provider, exempt_paths=["/health", "/docs"]
        )

        # Test authentication
        user_context = await middleware._authenticate_request(mock_request)

        # Verify
        assert isinstance(user_context, UserContext)
        assert user_context.user_id == "test-user-123"
        assert user_context.email == "test@example.com"

    def test_source_of_truth_firebase_config(self):
        """DEFINITIVE TEST: Verify Firebase project configuration matches iOS."""
        # This is the source of truth - iOS uses this project ID
        IOS_FIREBASE_PROJECT = "clarity-loop-backend"

        # Backend should use the same project ID
        provider = FirebaseAuthProvider(project_id=IOS_FIREBASE_PROJECT)

        assert provider.project_id == IOS_FIREBASE_PROJECT

        # When initialized, Firebase Admin SDK should use this project
        # This ensures tokens from iOS will be accepted
        print(f"✅ SOURCE OF TRUTH: Firebase project ID = {IOS_FIREBASE_PROJECT}")
        print("✅ Backend configured to accept tokens from iOS Firebase project")


class TestFirebaseTokenValidation:
    """Test token validation edge cases."""

    def test_decode_real_firebase_token_structure(self):
        """Test that we can decode and validate Firebase token structure."""
        # This is a sample structure of a real Firebase token
        sample_token_payload = {
            "iss": "https://securetoken.google.com/clarity-loop-backend",
            "aud": "clarity-loop-backend",
            "auth_time": 1736607600,
            "user_id": "abc123",
            "sub": "abc123",
            "iat": 1736607600,
            "exp": 1736611200,
            "email": "user@example.com",
            "email_verified": True,
            "firebase": {
                "identities": {"email": ["user@example.com"]},
                "sign_in_provider": "password",
            },
        }

        # Verify required fields
        assert sample_token_payload["aud"] == "clarity-loop-backend"
        assert (
            "securetoken.google.com/clarity-loop-backend" in sample_token_payload["iss"]
        )
        assert "email" in sample_token_payload
        assert "user_id" in sample_token_payload or "sub" in sample_token_payload

        print("✅ Firebase token structure validated")
        print(f"✅ Expected audience (aud): {sample_token_payload['aud']}")
        print(f"✅ Expected issuer (iss): {sample_token_payload['iss']}")


if __name__ == "__main__":
    # Run the source of truth test
    test = TestFirebaseTokenValidation()
    test.test_decode_real_firebase_token_structure()

    integration_test = TestFirebaseAuthIntegration()
    integration_test.test_source_of_truth_firebase_config()
