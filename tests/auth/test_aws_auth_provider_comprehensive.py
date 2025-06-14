"""Comprehensive tests for AWS Cognito Authentication Provider.

Tests critical production authentication flows including token verification,
user context creation, error handling, and role-based permissions.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from urllib.error import URLError

from botocore.exceptions import ClientError
from jose import JWTError
import pytest

from clarity.auth.aws_auth_provider import CognitoAuthProvider
from clarity.models.auth import AuthError, Permission, UserRole


class TestCognitoAuthProviderInitialization:
    """Test Cognito auth provider initialization."""

    def test_cognito_auth_provider_initialization_basic(self):
        """Test basic initialization of Cognito auth provider."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123", region="us-east-1"
        )

        assert provider.user_pool_id == "us-east-1_ABC123"
        assert provider.client_id == "client123"
        assert provider.region == "us-east-1"
        assert provider.users_table == "clarity_users"
        assert provider.cache_is_enabled is True
        assert not provider._initialized

    def test_cognito_auth_provider_initialization_with_dynamodb(self):
        """Test initialization with DynamoDB service."""
        mock_dynamodb = Mock()
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123",
            client_id="client123",
            region="us-west-2",
            dynamodb_service=mock_dynamodb,
        )

        assert provider.dynamodb_service is mock_dynamodb
        assert provider.region == "us-west-2"

    def test_cognito_auth_provider_initialization_with_middleware_config(self):
        """Test initialization with middleware configuration."""
        config = {
            "auth_provider_config": {
                "cache_enabled": False,
                "cache_ttl_seconds": 600,
                "cache_max_size": 2000,
            }
        }

        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123",
            client_id="client123",
            middleware_config=config,
        )

        assert provider.cache_is_enabled is False
        assert provider._token_cache_ttl_seconds == 600
        assert provider._token_cache_max_size == 2000

    @patch("boto3.client")
    async def test_cognito_auth_provider_initialize_success(
        self, mock_boto_client: MagicMock
    ) -> None:
        """Test successful initialization with AWS connection."""
        mock_cognito = Mock()
        mock_cognito.describe_user_pool.return_value = {
            "UserPool": {"Name": "TestUserPool"}
        }
        mock_boto_client.return_value = mock_cognito

        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )

        with patch.object(provider, "_get_jwks", return_value={"keys": []}):
            await provider.initialize()

        assert provider._initialized is True
        mock_cognito.describe_user_pool.assert_called_once_with(
            UserPoolId="us-east-1_ABC123"
        )

    @patch("boto3.client")
    async def test_cognito_auth_provider_initialize_failure(
        self, mock_boto_client: MagicMock
    ) -> None:
        """Test initialization failure handling."""
        mock_cognito = Mock()
        mock_cognito.describe_user_pool.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException"}}, "describe_user_pool"
        )
        mock_boto_client.return_value = mock_cognito

        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )

        with pytest.raises(
            RuntimeError, match="Could not initialize Cognito Auth Provider"
        ):
            await provider.initialize()

        assert not provider._initialized


class TestJWKSHandling:
    """Test JWKS (JSON Web Key Set) handling."""

    def test_jwks_url_construction(self):
        """Test JWKS URL is constructed correctly."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123", region="us-west-2"
        )

        expected_url = "https://cognito-idp.us-west-2.amazonaws.com/us-east-1_ABC123/.well-known/jwks.json"
        assert provider.jwks_url == expected_url

    @patch("urllib.request.urlopen")
    async def test_get_jwks_success(self, mock_urlopen: MagicMock) -> None:
        """Test successful JWKS retrieval."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            {
                "keys": [
                    {
                        "kid": "key1",
                        "kty": "RSA",
                        "use": "sig",
                        "n": "sample_n",
                        "e": "AQAB",
                    }
                ]
            }
        ).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response

        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )

        jwks = await provider._get_jwks()

        assert "keys" in jwks
        assert len(jwks["keys"]) == 1
        assert jwks["keys"][0]["kid"] == "key1"

    @patch("urllib.request.urlopen")
    async def test_get_jwks_caching(self, mock_urlopen: MagicMock) -> None:
        """Test JWKS caching functionality."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({"keys": []}).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response

        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )

        # First call
        jwks1 = await provider._get_jwks()
        # Second call should use cache
        jwks2 = await provider._get_jwks()

        assert jwks1 == jwks2
        # Should only call urlopen once due to caching
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    async def test_get_jwks_invalid_url_scheme(self, mock_urlopen: MagicMock) -> None:
        """Test JWKS retrieval with invalid URL scheme."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )
        provider.jwks_url = "http://insecure-url.com/jwks.json"  # HTTP instead of HTTPS

        with pytest.raises(ValueError, match="Invalid URL scheme"):
            await provider._get_jwks()

    @patch("urllib.request.urlopen")
    async def test_get_jwks_network_error_with_cache_fallback(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Test JWKS network error with cache fallback."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )

        # Set up existing cache
        provider._jwks_cache = {"keys": [{"kid": "cached_key"}]}
        provider._jwks_cache_time = time.time()

        # Mock network failure
        mock_urlopen.side_effect = URLError("Network error")

        # Should return cached JWKS
        jwks = await provider._get_jwks()
        assert jwks["keys"][0]["kid"] == "cached_key"


class TestTokenVerification:
    """Test token verification logic."""

    @patch("boto3.client")
    async def test_verify_token_success(self, mock_boto_client: MagicMock) -> None:
        """Test successful token verification."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )
        provider._initialized = True

        # Mock JWKS
        mock_jwks = {
            "keys": [
                {
                    "kid": "test_key_id",
                    "kty": "RSA",
                    "use": "sig",
                    "n": "sample_n",
                    "e": "AQAB",
                }
            ]
        }

        # Mock JWT payload
        mock_payload = {
            "sub": "user123",
            "email": "test@example.com",
            "email_verified": True,
            "custom": {"role": "patient"},
            "cognito:username": "testuser",
            "token_use": "id",
        }

        with (
            patch.object(provider, "_get_jwks", return_value=mock_jwks),
            patch(
                "jose.jwt.get_unverified_header", return_value={"kid": "test_key_id"}
            ),
            patch("jose.jwt.decode", return_value=mock_payload),
        ):
            result = await provider.verify_token("mock_token")

            assert result["user_id"] == "user123"
            assert result["email"] == "test@example.com"
            assert result["verified"] is True
            assert result["custom_claims"] == {"role": "patient"}

    async def test_verify_token_initializes_provider(self):
        """Test that verify_token initializes provider if not already initialized."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )

        mock_jwks = {"keys": [{"kid": "test_key_id"}]}
        mock_payload = {"sub": "user123", "email": "test@example.com"}

        with (
            patch.object(provider, "initialize") as mock_init,
            patch.object(provider, "_get_jwks", return_value=mock_jwks),
            patch(
                "jose.jwt.get_unverified_header", return_value={"kid": "test_key_id"}
            ),
            patch("jose.jwt.decode", return_value=mock_payload),
        ):
            provider._initialized = False
            await provider.verify_token("mock_token")

            mock_init.assert_called_once()

    @patch("boto3.client")
    async def test_verify_token_caching(self, mock_boto_client: MagicMock) -> None:
        """Test token verification caching."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )
        provider._initialized = True
        provider.cache_is_enabled = True

        # Pre-populate cache
        cached_user_data = {"user_id": "cached_user", "email": "cached@example.com"}
        provider._token_cache["cached_token"] = {
            "user_data": cached_user_data,
            "timestamp": time.time(),
        }

        result = await provider.verify_token("cached_token")

        assert result == cached_user_data

    async def test_verify_token_cache_disabled(self):
        """Test token verification when caching is disabled."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123",
            client_id="client123",
            middleware_config={"auth_provider_config": {"cache_enabled": False}},
        )
        provider._initialized = True

        mock_jwks = {"keys": [{"kid": "test_key_id"}]}
        mock_payload = {"sub": "user123", "email": "test@example.com"}

        with (
            patch.object(provider, "_get_jwks", return_value=mock_jwks),
            patch(
                "jose.jwt.get_unverified_header", return_value={"kid": "test_key_id"}
            ),
            patch("jose.jwt.decode", return_value=mock_payload),
        ):
            await provider.verify_token("test_token")

            # Token should not be cached
            assert "test_token" not in provider._token_cache

    async def test_verify_token_key_not_found(self):
        """Test token verification when key ID is not found in JWKS."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )
        provider._initialized = True

        mock_jwks = {"keys": [{"kid": "different_key_id"}]}

        with (
            patch.object(provider, "_get_jwks", return_value=mock_jwks),
            patch(
                "jose.jwt.get_unverified_header", return_value={"kid": "missing_key_id"}
            ),
            pytest.raises(AuthError, match="Unable to find appropriate key"),
        ):
            await provider.verify_token("invalid_token")

    async def test_verify_token_jwt_error(self):
        """Test token verification JWT decode error handling."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )
        provider._initialized = True

        mock_jwks = {"keys": [{"kid": "test_key_id"}]}

        with (
            patch.object(provider, "_get_jwks", return_value=mock_jwks),
            patch(
                "jose.jwt.get_unverified_header", return_value={"kid": "test_key_id"}
            ),
            patch("jose.jwt.decode", side_effect=JWTError("Invalid token")),
            pytest.raises(AuthError, match="Invalid Cognito token"),
        ):
            await provider.verify_token("invalid_token")

    async def test_verify_token_unexpected_error(self):
        """Test token verification unexpected error handling."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )
        provider._initialized = True

        with (
            patch.object(
                provider, "_get_jwks", side_effect=Exception("Unexpected error")
            ),
            pytest.raises(AuthError, match="An unexpected error occurred"),
        ):
            await provider.verify_token("test_token")


class TestTokenCacheManagement:
    """Test token cache management functionality."""

    def test_remove_expired_tokens_with_cache_disabled(self):
        """Test cache cleanup when caching is disabled."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123",
            client_id="client123",
            middleware_config={"auth_provider_config": {"cache_enabled": False}},
        )

        # Add some tokens to cache anyway
        provider._token_cache["token1"] = {"timestamp": time.time() - 1000}

        provider._remove_expired_tokens()

        # Should not remove anything when cache is disabled
        assert "token1" in provider._token_cache

    def test_remove_expired_tokens_removes_old_tokens(self):
        """Test that expired tokens are removed from cache."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )
        provider._token_cache_ttl_seconds = 300  # 5 minutes

        current_time = time.time()
        # Add expired token
        provider._token_cache["expired_token"] = {
            "user_data": {"user_id": "expired"},
            "timestamp": current_time - 600,  # 10 minutes ago
        }
        # Add valid token
        provider._token_cache["valid_token"] = {
            "user_data": {"user_id": "valid"},
            "timestamp": current_time - 100,  # 1.6 minutes ago
        }

        provider._remove_expired_tokens()

        assert "expired_token" not in provider._token_cache
        assert "valid_token" in provider._token_cache


class TestUserInfoRetrieval:
    """Test user information retrieval from Cognito."""

    @patch("boto3.client")
    async def test_get_user_info_by_username_success(
        self, mock_boto_client: MagicMock
    ) -> None:
        """Test successful user info retrieval by username."""
        mock_cognito = Mock()
        mock_cognito.admin_get_user.return_value = {
            "UserAttributes": [
                {"Name": "sub", "Value": "user123"},
                {"Name": "email", "Value": "test@example.com"},
                {"Name": "email_verified", "Value": "true"},
                {"Name": "custom:role", "Value": "admin"},
            ]
        }
        mock_boto_client.return_value = mock_cognito

        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )
        provider._initialized = True

        result = await provider.get_user_info("testuser")

        assert result["user_id"] == "user123"
        assert result["email"] == "test@example.com"
        assert result["verified"] is True
        assert "admin" in result["roles"]

    @patch("boto3.client")
    async def test_get_user_info_by_email_fallback(
        self, mock_boto_client: MagicMock
    ) -> None:
        """Test user info retrieval fallback to email search."""
        mock_cognito = Mock()
        # First call (by username) fails
        mock_cognito.admin_get_user.side_effect = ClientError(
            {"Error": {"Code": "UserNotFoundException"}}, "admin_get_user"
        )
        # Second call (by email) succeeds
        mock_cognito.list_users.return_value = {
            "Users": [
                {
                    "Attributes": [
                        {"Name": "sub", "Value": "user123"},
                        {"Name": "email", "Value": "test@example.com"},
                        {"Name": "custom:role", "Value": "clinician"},
                    ]
                }
            ]
        }
        mock_boto_client.return_value = mock_cognito

        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )
        provider._initialized = True

        result = await provider.get_user_info("test@example.com")

        assert result["user_id"] == "user123"
        assert "clinician" in result["roles"]

    @patch("boto3.client")
    async def test_get_user_info_user_not_found(
        self, mock_boto_client: MagicMock
    ) -> None:
        """Test user info retrieval when user is not found."""
        mock_cognito = Mock()
        mock_cognito.admin_get_user.side_effect = ClientError(
            {"Error": {"Code": "UserNotFoundException"}}, "admin_get_user"
        )
        mock_cognito.list_users.return_value = {"Users": []}
        mock_boto_client.return_value = mock_cognito

        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )
        provider._initialized = True

        result = await provider.get_user_info("nonexistent@example.com")

        assert result is None

    async def test_get_user_info_initializes_provider(self):
        """Test that get_user_info initializes provider if needed."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )

        with (
            patch.object(provider, "initialize") as mock_init,
            patch.object(
                provider.cognito_client,
                "admin_get_user",
                return_value={"UserAttributes": []},
            ),
        ):
            provider._initialized = False
            await provider.get_user_info("testuser")

            mock_init.assert_called_once()


class TestUserContextCreation:
    """Test user context creation and management."""

    def test_create_basic_user_context_patient_role(self):
        """Test creation of basic user context with patient role."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )

        user_info = {
            "user_id": "user123",
            "email": "patient@example.com",
            "verified": True,
            "custom_claims": {"role": "patient"},
        }

        context = provider._create_basic_user_context(user_info)

        assert context.user_id == "user123"
        assert context.email == "patient@example.com"
        assert context.role == UserRole.PATIENT
        assert context.is_verified is True
        assert Permission.READ_OWN_DATA in context.permissions
        assert Permission.WRITE_OWN_DATA in context.permissions
        assert Permission.READ_PATIENT_DATA not in context.permissions

    def test_create_basic_user_context_clinician_role(self):
        """Test creation of basic user context with clinician role."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )

        user_info = {
            "user_id": "clinician123",
            "email": "clinician@example.com",
            "verified": True,
            "custom_claims": {"role": "clinician"},
        }

        context = provider._create_basic_user_context(user_info)

        assert context.role == UserRole.CLINICIAN
        assert Permission.READ_OWN_DATA in context.permissions
        assert Permission.WRITE_OWN_DATA in context.permissions
        assert Permission.READ_PATIENT_DATA in context.permissions
        assert Permission.WRITE_PATIENT_DATA in context.permissions
        assert Permission.SYSTEM_ADMIN not in context.permissions

    def test_create_basic_user_context_admin_role(self):
        """Test creation of basic user context with admin role."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )

        user_info = {
            "user_id": "admin123",
            "email": "admin@example.com",
            "verified": True,
            "custom_claims": {"role": "admin"},
        }

        context = provider._create_basic_user_context(user_info)

        assert context.role == UserRole.ADMIN
        assert Permission.SYSTEM_ADMIN in context.permissions
        assert Permission.MANAGE_USERS in context.permissions
        assert Permission.READ_PATIENT_DATA in context.permissions
        assert Permission.READ_ANONYMIZED_DATA in context.permissions

    def test_create_basic_user_context_default_role(self):
        """Test creation of basic user context with default role."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )

        user_info = {
            "user_id": "user123",
            "email": "user@example.com",
            "verified": False,
            "custom_claims": {},  # No role specified
        }

        context = provider._create_basic_user_context(user_info)

        assert context.role == UserRole.PATIENT  # Default role
        assert context.is_verified is False

    async def test_get_or_create_user_context_without_dynamodb(self):
        """Test user context creation when no DynamoDB service is available."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123",
            client_id="client123",
            dynamodb_service=None,
        )

        cognito_user_info = {
            "user_id": "user123",
            "email": "test@example.com",
            "verified": True,
            "custom_claims": {"role": "patient"},
        }

        context = await provider.get_or_create_user_context(cognito_user_info)

        assert context.user_id == "user123"
        assert context.role == UserRole.PATIENT

    async def test_get_or_create_user_context_with_existing_dynamodb_user(self):
        """Test user context creation with existing DynamoDB user."""
        mock_dynamodb = AsyncMock()
        mock_dynamodb.get_item.return_value = {
            "user_id": "user123",
            "email": "test@example.com",
            "role": "patient",
            "status": "active",
        }

        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123",
            client_id="client123",
            dynamodb_service=mock_dynamodb,
        )

        cognito_user_info = {
            "user_id": "user123",
            "email": "test@example.com",
            "verified": True,
            "custom_claims": {},
        }

        context = await provider.get_or_create_user_context(cognito_user_info)

        assert context.user_id == "user123"
        mock_dynamodb.update_item.assert_called_once()  # Should update last login

    async def test_get_or_create_user_context_creates_new_user(self):
        """Test user context creation when creating new DynamoDB user."""
        mock_dynamodb = AsyncMock()
        mock_dynamodb.get_item.return_value = None  # User doesn't exist

        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123",
            client_id="client123",
            dynamodb_service=mock_dynamodb,
        )

        cognito_user_info = {
            "user_id": "newuser123",
            "email": "new@example.com",
            "verified": True,
            "custom_claims": {"given_name": "John", "family_name": "Doe"},
        }

        with patch.object(
            provider,
            "_create_user_record",
            return_value={
                "user_id": "newuser123",
                "email": "new@example.com",
                "role": "patient",
            },
        ) as mock_create:
            context = await provider.get_or_create_user_context(cognito_user_info)

            assert context.user_id == "newuser123"
            mock_create.assert_called_once_with(cognito_user_info)

    async def test_get_or_create_user_context_fallback_on_error(self):
        """Test user context creation fallback when DynamoDB fails."""
        mock_dynamodb = AsyncMock()
        mock_dynamodb.get_item.side_effect = Exception("DynamoDB error")

        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123",
            client_id="client123",
            dynamodb_service=mock_dynamodb,
        )

        cognito_user_info = {
            "user_id": "user123",
            "email": "test@example.com",
            "verified": True,
            "custom_claims": {"role": "patient"},
        }

        context = await provider.get_or_create_user_context(cognito_user_info)

        # Should fall back to basic context creation
        assert context.user_id == "user123"
        assert context.role == UserRole.PATIENT


class TestUserRecordCreation:
    """Test DynamoDB user record creation."""

    async def test_create_user_record_success(self):
        """Test successful user record creation in DynamoDB."""
        mock_dynamodb = AsyncMock()

        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123",
            client_id="client123",
            dynamodb_service=mock_dynamodb,
        )

        cognito_user_info = {
            "user_id": "user123",
            "email": "test@example.com",
            "verified": True,
            "custom_claims": {
                "given_name": "John",
                "family_name": "Doe",
                "role": "clinician",
            },
        }

        result = await provider._create_user_record(cognito_user_info)

        assert result["user_id"] == "user123"
        assert result["email"] == "test@example.com"
        assert result["first_name"] == "John"
        assert result["last_name"] == "Doe"
        assert result["role"] == UserRole.CLINICIAN.value
        assert result["auth_provider"] == "COGNITO"
        assert result["login_count"] == 1

        mock_dynamodb.put_item.assert_called_once()

    def test_create_user_context_from_db_success(self):
        """Test user context creation from database record."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )

        user_data = {
            "user_id": "user123",
            "email": "test@example.com",
            "first_name": "John",
            "last_name": "Doe",
            "role": "clinician",
            "status": "active",
            "email_verified": True,
            "created_at": "2024-01-01T00:00:00Z",
            "last_login": "2024-01-01T12:00:00Z",
            "custom_claims": {"department": "cardiology"},
        }

        cognito_info = {"user_id": "user123"}

        context = provider._create_user_context_from_db(user_data, cognito_info)

        assert context.user_id == "user123"
        assert context.email == "test@example.com"
        assert context.role == UserRole.CLINICIAN
        assert context.custom_claims == {"department": "cardiology"}

    def test_create_user_context_from_db_invalid_role(self):
        """Test user context creation with invalid role defaults to patient."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )

        user_data = {
            "user_id": "user123",
            "email": "test@example.com",
            "role": "invalid_role",  # Invalid role
            "status": "active",
        }

        context = provider._create_user_context_from_db(user_data, {})

        assert context.role == UserRole.PATIENT  # Should default to patient


class TestCleanup:
    """Test provider cleanup functionality."""

    async def test_cleanup_method_exists(self):
        """Test that cleanup method exists and can be called."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )

        # Should not raise an exception
        await provider.cleanup()


class TestProductionIntegrationScenarios:
    """Test realistic production integration scenarios."""

    async def test_full_authentication_flow_with_dynamodb(self):
        """Test complete authentication flow with DynamoDB integration."""
        mock_dynamodb = AsyncMock()
        mock_dynamodb.get_item.return_value = None  # New user

        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123",
            client_id="client123",
            dynamodb_service=mock_dynamodb,
        )
        provider._initialized = True

        # Mock successful token verification
        mock_jwks = {"keys": [{"kid": "test_key"}]}
        mock_payload = {
            "sub": "user123",
            "email": "user@example.com",
            "email_verified": True,
            "custom": {"role": "patient", "given_name": "John"},
            "cognito:username": "john_doe",
            "token_use": "id",
        }

        with (
            patch.object(provider, "_get_jwks", return_value=mock_jwks),
            patch("jose.jwt.get_unverified_header", return_value={"kid": "test_key"}),
            patch("jose.jwt.decode", return_value=mock_payload),
            patch.object(
                provider,
                "_create_user_record",
                return_value={
                    "user_id": "user123",
                    "email": "user@example.com",
                    "role": "patient",
                },
            ),
        ):
            # Verify token
            user_info = await provider.verify_token("valid_jwt_token")
            assert user_info["user_id"] == "user123"

            # Create user context
            context = await provider.get_or_create_user_context(user_info)
            assert context.user_id == "user123"
            assert context.role == UserRole.PATIENT

    async def test_authentication_error_handling_chain(self):
        """Test proper error handling throughout authentication chain."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )
        provider._initialized = True

        # Test JWKS failure
        with (
            patch.object(provider, "_get_jwks", side_effect=Exception("JWKS failed")),
            pytest.raises(AuthError, match="An unexpected error occurred"),
        ):
            await provider.verify_token("any_token")

    async def test_concurrent_token_verification(self):
        """Test concurrent token verification doesn't break caching."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )
        provider._initialized = True

        mock_jwks = {"keys": [{"kid": "test_key"}]}
        mock_payload = {"sub": "user123", "email": "test@example.com"}

        with (
            patch.object(provider, "_get_jwks", return_value=mock_jwks),
            patch("jose.jwt.get_unverified_header", return_value={"kid": "test_key"}),
            patch("jose.jwt.decode", return_value=mock_payload),
        ):
            # Simulate concurrent calls
            tasks = [
                provider.verify_token("token1"),
                provider.verify_token("token1"),  # Same token
                provider.verify_token("token2"),
            ]

            results = await asyncio.gather(*tasks)

            # All should succeed
            assert all(result["user_id"] == "user123" for result in results)
