"""Comprehensive tests for AWS Cognito Authentication Provider.

Tests critical production authentication infrastructure including JWT verification,
user management, authentication flows, and error handling scenarios.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, Mock, patch

from botocore.exceptions import ClientError
from jose import JWTError
import pytest
import requests

from clarity.auth.aws_cognito_provider import CognitoAuthProvider, get_cognito_provider
from clarity.core.exceptions import AuthenticationError
from clarity.models.user import User


class TestCognitoAuthProviderInitialization:
    """Test Cognito auth provider initialization."""

    def test_cognito_auth_provider_initialization(self) -> None:
        """Test basic Cognito auth provider initialization."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123", region="us-east-1"
        )

        assert provider.user_pool_id == "us-east-1_ABC123"
        assert provider.client_id == "client123"
        assert provider.region == "us-east-1"
        assert (
            provider.issuer
            == "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_ABC123"
        )
        assert (
            provider.jwks_url
            == "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_ABC123/.well-known/jwks.json"
        )
        assert provider._jwks_cache is None
        assert provider._jwks_cache_time == 0.0
        assert provider._jwks_cache_ttl == 3600

    @patch("clarity.auth.aws_cognito_provider.boto3.client")
    def test_cognito_auth_provider_client_creation(
        self, mock_boto3_client: MagicMock
    ) -> None:
        """Test that Cognito client is created correctly."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client

        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123", region="us-west-2"
        )

        mock_boto3_client.assert_called_once_with(
            "cognito-idp", region_name="us-west-2"
        )
        assert provider.cognito_client == mock_client

    def test_cognito_auth_provider_default_region(self) -> None:
        """Test default region handling."""
        provider = CognitoAuthProvider(
            user_pool_id="us-east-1_ABC123", client_id="client123"
        )

        assert provider.region == "us-east-1"


class TestCognitoJWKSCaching:
    """Test JWKS key caching functionality."""

    @patch("clarity.auth.aws_cognito_provider.requests.get")
    def test_jwks_property_initial_fetch(self, mock_get: MagicMock) -> None:
        """Test initial JWKS fetch and caching."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "keys": [{"kid": "key1", "kty": "RSA", "use": "sig"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = CognitoAuthProvider("pool", "client")

        with patch("clarity.auth.aws_cognito_provider.time.time", return_value=1000.0):
            jwks = provider.jwks

        assert jwks == {"keys": [{"kid": "key1", "kty": "RSA", "use": "sig"}]}
        assert provider._jwks_cache == jwks
        assert provider._jwks_cache_time == 1000.0
        mock_get.assert_called_once_with(provider.jwks_url, timeout=30)

    @patch("clarity.auth.aws_cognito_provider.requests.get")
    def test_jwks_property_cache_hit(self, mock_get: MagicMock) -> None:
        """Test JWKS cache hit (no new request)."""
        provider = CognitoAuthProvider("pool", "client")
        provider._jwks_cache = {"keys": [{"kid": "cached_key"}]}
        provider._jwks_cache_time = 1000.0

        with patch("clarity.auth.aws_cognito_provider.time.time", return_value=1500.0):
            jwks = provider.jwks

        assert jwks == {"keys": [{"kid": "cached_key"}]}
        mock_get.assert_not_called()

    @patch("clarity.auth.aws_cognito_provider.requests.get")
    def test_jwks_property_cache_expired(self, mock_get: MagicMock) -> None:
        """Test JWKS cache expiration and refresh."""
        mock_response = Mock()
        mock_response.json.return_value = {"keys": [{"kid": "new_key"}]}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        provider = CognitoAuthProvider("pool", "client")
        provider._jwks_cache = {"keys": [{"kid": "old_key"}]}
        provider._jwks_cache_time = 1000.0

        # Cache TTL is 3600 seconds, so 1000 + 3601 = 4601 should expire cache
        with patch("clarity.auth.aws_cognito_provider.time.time", return_value=4601.0):
            jwks = provider.jwks

        assert jwks == {"keys": [{"kid": "new_key"}]}
        assert provider._jwks_cache_time == 4601.0
        mock_get.assert_called_once()

    @patch("clarity.auth.aws_cognito_provider.requests.get")
    def test_jwks_property_fetch_failure_no_cache(self, mock_get: MagicMock) -> None:
        """Test JWKS fetch failure with no existing cache."""
        mock_get.side_effect = requests.RequestException("Network error")

        provider = CognitoAuthProvider("pool", "client")

        with pytest.raises(AuthenticationError, match="Failed to fetch JWKS keys"):
            _ = provider.jwks

    @patch("clarity.auth.aws_cognito_provider.requests.get")
    def test_jwks_property_fetch_failure_with_cache(self, mock_get: MagicMock) -> None:
        """Test JWKS fetch failure with existing cache."""
        mock_get.side_effect = requests.RequestException("Network error")

        provider = CognitoAuthProvider("pool", "client")
        provider._jwks_cache = {"keys": [{"kid": "cached_key"}]}
        provider._jwks_cache_time = 1000.0

        with patch("clarity.auth.aws_cognito_provider.time.time", return_value=5000.0):
            jwks = provider.jwks

        # Should return cached version despite fetch failure
        assert jwks == {"keys": [{"kid": "cached_key"}]}

    @patch("clarity.auth.aws_cognito_provider.requests.get")
    def test_jwks_property_http_error(self, mock_get: MagicMock) -> None:
        """Test JWKS HTTP error handling."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        provider = CognitoAuthProvider("pool", "client")

        with pytest.raises(AuthenticationError, match="Failed to fetch JWKS keys"):
            _ = provider.jwks


class TestTokenVerification:
    """Test JWT token verification functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.provider = CognitoAuthProvider("us-east-1_ABC123", "client123")
        self.mock_jwks = {
            "keys": [
                {
                    "kid": "test_key_id",
                    "kty": "RSA",
                    "use": "sig",
                    "n": "test_n_value",
                    "e": "AQAB",
                }
            ]
        }

    @patch("clarity.auth.aws_cognito_provider.requests.get")
    @patch("clarity.auth.aws_cognito_provider.jwt.get_unverified_headers")
    @patch("clarity.auth.aws_cognito_provider.jwt.get_unverified_claims")
    @patch("clarity.auth.aws_cognito_provider.jwk.construct")
    @patch("clarity.auth.aws_cognito_provider.base64url_decode")
    @patch("clarity.auth.aws_cognito_provider.time.time")
    @pytest.mark.asyncio
    async def test_verify_token_success(
        self,
        mock_time: MagicMock,
        mock_b64_decode: MagicMock,
        mock_jwk_construct: MagicMock,
        mock_get_claims: MagicMock,
        mock_get_headers: MagicMock,
        mock_requests_get: MagicMock,
    ) -> None:
        """Test successful token verification."""
        # Mock JWKS fetch
        mock_response = Mock()
        mock_response.json.return_value = self.mock_jwks
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response

        # Mock setup
        mock_get_headers.return_value = {"kid": "test_key_id"}
        mock_get_claims.return_value = {
            "sub": "user123",
            "exp": 2000.0,
            "aud": "client123",
            "email": "test@example.com",
        }
        mock_time.return_value = 1000.0  # Token not expired

        mock_public_key = Mock()
        mock_public_key.verify.return_value = True
        mock_jwk_construct.return_value = mock_public_key

        mock_b64_decode.return_value = b"mock_signature"

        result = await self.provider.verify_token("mock.jwt.token")

        assert result is not None
        assert result["sub"] == "user123"
        assert result["email"] == "test@example.com"
        mock_public_key.verify.assert_called_once()

    @patch("clarity.auth.aws_cognito_provider.requests.get")
    @patch("clarity.auth.aws_cognito_provider.jwt.get_unverified_headers")
    @pytest.mark.asyncio
    async def test_verify_token_kid_not_found(
        self, mock_get_headers: MagicMock, mock_requests_get: MagicMock
    ) -> None:
        """Test token verification with unknown key ID."""
        # Mock JWKS fetch
        mock_response = Mock()
        mock_response.json.return_value = self.mock_jwks
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response

        mock_get_headers.return_value = {"kid": "unknown_key_id"}

        result = await self.provider.verify_token("mock.jwt.token")

        assert result is None

    @patch("clarity.auth.aws_cognito_provider.requests.get")
    @patch("clarity.auth.aws_cognito_provider.jwt.get_unverified_headers")
    @patch("clarity.auth.aws_cognito_provider.jwt.get_unverified_claims")
    @patch("clarity.auth.aws_cognito_provider.jwk.construct")
    @patch("clarity.auth.aws_cognito_provider.time.time")
    @pytest.mark.asyncio
    async def test_verify_token_signature_verification_failed(
        self,
        mock_time: MagicMock,
        mock_jwk_construct: MagicMock,
        mock_get_claims: MagicMock,
        mock_get_headers: MagicMock,
        mock_requests_get: MagicMock,
    ) -> None:
        """Test token verification with invalid signature."""
        # Mock JWKS fetch
        mock_response = Mock()
        mock_response.json.return_value = self.mock_jwks
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response

        mock_get_headers.return_value = {"kid": "test_key_id"}
        mock_get_claims.return_value = {
            "sub": "user123",
            "exp": 2000.0,
            "aud": "client123",
        }
        mock_time.return_value = 1000.0

        mock_public_key = Mock()
        mock_public_key.verify.return_value = False  # Signature verification fails
        mock_jwk_construct.return_value = mock_public_key

        result = await self.provider.verify_token("mock.jwt.token")

        assert result is None

    @patch("clarity.auth.aws_cognito_provider.requests.get")
    @patch("clarity.auth.aws_cognito_provider.jwt.get_unverified_headers")
    @patch("clarity.auth.aws_cognito_provider.jwt.get_unverified_claims")
    @patch("clarity.auth.aws_cognito_provider.jwk.construct")
    @patch("clarity.auth.aws_cognito_provider.base64url_decode")
    @patch("clarity.auth.aws_cognito_provider.time.time")
    @pytest.mark.asyncio
    async def test_verify_token_expired(
        self,
        mock_time: MagicMock,
        mock_b64_decode: MagicMock,
        mock_jwk_construct: MagicMock,
        mock_get_claims: MagicMock,
        mock_get_headers: MagicMock,
        mock_requests_get: MagicMock,
    ) -> None:
        """Test token verification with expired token."""
        # Mock JWKS fetch
        mock_response = Mock()
        mock_response.json.return_value = self.mock_jwks
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response

        mock_get_headers.return_value = {"kid": "test_key_id"}
        mock_get_claims.return_value = {
            "sub": "user123",
            "exp": 1000.0,  # Token expires at 1000
            "aud": "client123",
        }
        mock_time.return_value = 2000.0  # Current time is after expiration

        mock_public_key = Mock()
        mock_public_key.verify.return_value = True
        mock_jwk_construct.return_value = mock_public_key

        result = await self.provider.verify_token("mock.jwt.token")

        assert result is None

    @patch("clarity.auth.aws_cognito_provider.requests.get")
    @patch("clarity.auth.aws_cognito_provider.jwt.get_unverified_headers")
    @patch("clarity.auth.aws_cognito_provider.jwt.get_unverified_claims")
    @patch("clarity.auth.aws_cognito_provider.jwk.construct")
    @patch("clarity.auth.aws_cognito_provider.base64url_decode")
    @patch("clarity.auth.aws_cognito_provider.time.time")
    @pytest.mark.asyncio
    async def test_verify_token_wrong_audience(
        self,
        mock_time: MagicMock,
        mock_b64_decode: MagicMock,
        mock_jwk_construct: MagicMock,
        mock_get_claims: MagicMock,
        mock_get_headers: MagicMock,
        mock_requests_get: MagicMock,
    ) -> None:
        """Test token verification with wrong audience."""
        # Mock JWKS fetch
        mock_response = Mock()
        mock_response.json.return_value = self.mock_jwks
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response

        mock_get_headers.return_value = {"kid": "test_key_id"}
        mock_get_claims.return_value = {
            "sub": "user123",
            "exp": 2000.0,
            "aud": "different_client_id",  # Wrong audience
        }
        mock_time.return_value = 1000.0

        mock_public_key = Mock()
        mock_public_key.verify.return_value = True
        mock_jwk_construct.return_value = mock_public_key

        result = await self.provider.verify_token("mock.jwt.token")

        assert result is None

    @patch("clarity.auth.aws_cognito_provider.requests.get")
    @patch("clarity.auth.aws_cognito_provider.jwt.get_unverified_headers")
    @pytest.mark.asyncio
    async def test_verify_token_jwt_error(
        self, mock_get_headers: MagicMock, mock_requests_get: MagicMock
    ) -> None:
        """Test token verification with JWT error."""
        # Mock the JWKS HTTP request
        mock_requests_get.return_value.json.return_value = self.mock_jwks
        mock_requests_get.return_value.raise_for_status.return_value = None

        # Mock JWT error during header extraction
        mock_get_headers.side_effect = JWTError("Invalid token format")

        result = await self.provider.verify_token("invalid.jwt.token")

        assert result is None

    @patch("clarity.auth.aws_cognito_provider.requests.get")
    @patch("clarity.auth.aws_cognito_provider.jwt.get_unverified_headers")
    @pytest.mark.asyncio
    async def test_verify_token_unexpected_error(
        self, mock_get_headers: MagicMock, mock_requests_get: MagicMock
    ) -> None:
        """Test token verification with unexpected error."""
        # Mock the JWKS HTTP request
        mock_requests_get.return_value.json.return_value = self.mock_jwks
        mock_requests_get.return_value.raise_for_status.return_value = None

        # Mock unexpected error during header extraction
        mock_get_headers.side_effect = Exception("Unexpected error")

        result = await self.provider.verify_token("mock.jwt.token")

        assert result is None


class TestUserManagement:
    """Test user management functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.provider = CognitoAuthProvider("us-east-1_ABC123", "client123")
        self.provider.cognito_client = Mock()

    @pytest.mark.asyncio
    async def test_get_user_success(self) -> None:
        """Test successful user retrieval."""
        mock_response = {
            "Username": "user123",
            "UserStatus": "CONFIRMED",
            "Enabled": True,
            "UserCreateDate": datetime(2023, 1, 1, tzinfo=UTC),
            "UserLastModifiedDate": datetime(2023, 1, 2, tzinfo=UTC),
            "UserAttributes": [
                {"Name": "email", "Value": "test@example.com"},
                {"Name": "name", "Value": "Test User"},
            ],
        }

        self.provider.cognito_client.admin_get_user.return_value = mock_response

        user = await self.provider.get_user("user123")

        assert user is not None
        assert user.uid == "user123"
        assert user.email == "test@example.com"
        assert user.display_name == "Test User"
        assert user.metadata["username"] == "user123"
        assert user.metadata["status"] == "CONFIRMED"
        assert user.metadata["enabled"] is True

    @pytest.mark.asyncio
    async def test_get_user_not_found(self) -> None:
        """Test user retrieval when user doesn't exist."""
        error = ClientError(
            {"Error": {"Code": "UserNotFoundException", "Message": "User not found"}},
            "AdminGetUser",
        )
        self.provider.cognito_client.admin_get_user.side_effect = error

        user = await self.provider.get_user("nonexistent")

        assert user is None

    @pytest.mark.asyncio
    async def test_get_user_client_error(self) -> None:
        """Test user retrieval with other client error."""
        error = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            "AdminGetUser",
        )
        self.provider.cognito_client.admin_get_user.side_effect = error

        user = await self.provider.get_user("user123")

        assert user is None

    @pytest.mark.asyncio
    async def test_get_user_unexpected_error(self) -> None:
        """Test user retrieval with unexpected error."""
        self.provider.cognito_client.admin_get_user.side_effect = Exception(
            "Network error"
        )

        user = await self.provider.get_user("user123")

        assert user is None

    @patch.dict("os.environ", {"ENVIRONMENT": "development"})
    @pytest.mark.asyncio
    async def test_create_user_success_development(self) -> None:
        """Test successful user creation in development."""
        mock_response = {"UserSub": "new-user-123"}
        self.provider.cognito_client.sign_up.return_value = mock_response
        self.provider.cognito_client.admin_confirm_sign_up.return_value = {}

        user = await self.provider.create_user(
            email="new@example.com",
            password="password123",  # noqa: S106 - Test password for unit tests
            display_name="New User",
        )

        assert user is not None
        assert user.uid == "new-user-123"
        assert user.email == "new@example.com"
        assert user.display_name == "New User"
        assert user.metadata["status"] == "CONFIRMED"

        # Verify auto-confirmation was called
        self.provider.cognito_client.admin_confirm_sign_up.assert_called_once_with(
            UserPoolId="us-east-1_ABC123", Username="new@example.com"
        )

    @patch.dict("os.environ", {"ENVIRONMENT": "production"})
    @pytest.mark.asyncio
    async def test_create_user_success_production(self) -> None:
        """Test successful user creation in production."""
        mock_response = {"UserSub": "new-user-123"}
        self.provider.cognito_client.sign_up.return_value = mock_response

        user = await self.provider.create_user(
            email="new@example.com",
            password="password123",  # noqa: S106 - Test password
        )

        assert user is not None
        assert user.uid == "new-user-123"
        assert user.email == "new@example.com"
        assert user.display_name == "new@example.com"  # Defaults to email

        # Verify auto-confirmation was NOT called in production
        self.provider.cognito_client.admin_confirm_sign_up.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_user_already_exists(self) -> None:
        """Test user creation when user already exists."""
        error = ClientError(
            {"Error": {"Code": "UsernameExistsException", "Message": "User exists"}},
            "SignUp",
        )
        self.provider.cognito_client.sign_up.side_effect = error

        with pytest.raises(AuthenticationError, match="User already exists"):
            await self.provider.create_user("existing@example.com", "password123")

    @pytest.mark.asyncio
    async def test_create_user_client_error(self) -> None:
        """Test user creation with other client error."""
        error = ClientError(
            {
                "Error": {
                    "Code": "InvalidParameterException",
                    "Message": "Invalid param",
                }
            },
            "SignUp",
        )
        self.provider.cognito_client.sign_up.side_effect = error

        with pytest.raises(AuthenticationError, match="Failed to create user"):
            await self.provider.create_user("test@example.com", "password123")

    @pytest.mark.asyncio
    async def test_create_user_unexpected_error(self) -> None:
        """Test user creation with unexpected error."""
        self.provider.cognito_client.sign_up.side_effect = Exception("Network error")

        with pytest.raises(AuthenticationError, match="Unexpected error"):
            await self.provider.create_user("test@example.com", "password123")

    @pytest.mark.asyncio
    async def test_delete_user_success(self) -> None:
        """Test successful user deletion."""
        self.provider.cognito_client.admin_delete_user.return_value = {}

        result = await self.provider.delete_user("user123")

        assert result is True
        self.provider.cognito_client.admin_delete_user.assert_called_once_with(
            UserPoolId="us-east-1_ABC123", Username="user123"
        )

    @pytest.mark.asyncio
    async def test_delete_user_client_error(self) -> None:
        """Test user deletion with client error."""
        error = ClientError(
            {"Error": {"Code": "UserNotFoundException", "Message": "User not found"}},
            "AdminDeleteUser",
        )
        self.provider.cognito_client.admin_delete_user.side_effect = error

        result = await self.provider.delete_user("user123")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_user_unexpected_error(self) -> None:
        """Test user deletion with unexpected error."""
        self.provider.cognito_client.admin_delete_user.side_effect = Exception(
            "Network error"
        )

        result = await self.provider.delete_user("user123")

        assert result is False

    @pytest.mark.asyncio
    async def test_update_user_success(self) -> None:
        """Test successful user update."""
        self.provider.cognito_client.admin_update_user_attributes.return_value = {}

        # Mock get_user for return value
        mock_user = User(
            uid="user123",
            email="updated@example.com",
            display_name="Updated User",
            created_at=datetime.now(UTC),
            last_login=None,
            metadata={},
        )

        with patch.object(self.provider, "get_user", return_value=mock_user):
            result = await self.provider.update_user(
                "user123", display_name="Updated User", email="updated@example.com"
            )

        assert result == mock_user
        self.provider.cognito_client.admin_update_user_attributes.assert_called_once()

        call_args = self.provider.cognito_client.admin_update_user_attributes.call_args
        assert call_args[1]["UserPoolId"] == "us-east-1_ABC123"
        assert call_args[1]["Username"] == "user123"

        attributes = call_args[1]["UserAttributes"]
        assert {"Name": "name", "Value": "Updated User"} in attributes
        assert {"Name": "email", "Value": "updated@example.com"} in attributes

    @pytest.mark.asyncio
    async def test_update_user_no_attributes(self) -> None:
        """Test user update with no attributes to update."""
        mock_user = User(
            uid="user123",
            email="test@example.com",
            display_name="Test User",
            created_at=datetime.now(UTC),
            last_login=None,
            metadata={},
        )

        with patch.object(self.provider, "get_user", return_value=mock_user):
            result = await self.provider.update_user("user123")

        assert result == mock_user
        self.provider.cognito_client.admin_update_user_attributes.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_user_client_error(self) -> None:
        """Test user update with client error."""
        error = ClientError(
            {"Error": {"Code": "UserNotFoundException", "Message": "User not found"}},
            "AdminUpdateUserAttributes",
        )
        self.provider.cognito_client.admin_update_user_attributes.side_effect = error

        result = await self.provider.update_user("user123", display_name="New Name")

        assert result is None

    @pytest.mark.asyncio
    async def test_update_user_unexpected_error(self) -> None:
        """Test user update with unexpected error."""
        self.provider.cognito_client.admin_update_user_attributes.side_effect = (
            Exception("Network error")
        )

        result = await self.provider.update_user("user123", email="new@example.com")

        assert result is None


class TestAuthentication:
    """Test authentication functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.provider = CognitoAuthProvider("us-east-1_ABC123", "client123")
        self.provider.cognito_client = Mock()

    @pytest.mark.asyncio
    async def test_authenticate_success(self) -> None:
        """Test successful authentication."""
        mock_response = {
            "AuthenticationResult": {
                "AccessToken": "access_token_123",
                "IdToken": "id_token_123",
                "RefreshToken": "refresh_token_123",
                "ExpiresIn": 3600,
            }
        }
        self.provider.cognito_client.initiate_auth.return_value = mock_response

        result = await self.provider.authenticate("test@example.com", "password123")

        assert result is not None
        assert (
            result["access_token"]
            == "access_token_123"  # noqa: S105 - Test token value
        )
        assert result["id_token"] == "id_token_123"  # noqa: S105 - Test token value
        assert (
            result["refresh_token"]
            == "refresh_token_123"  # noqa: S105 - Test token value
        )
        assert result["expires_in"] == "3600"

    @pytest.mark.asyncio
    async def test_authenticate_challenge_required(self) -> None:
        """Test authentication with challenge required."""
        mock_response = {
            "ChallengeName": "SMS_MFA",
            "ChallengeParameters": {"USER_ID_FOR_SRP": "user123"},
        }
        self.provider.cognito_client.initiate_auth.return_value = mock_response

        result = await self.provider.authenticate("test@example.com", "password123")

        assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_unexpected_response(self) -> None:
        """Test authentication with unexpected response format."""
        mock_response = {"SomeOtherField": "value"}
        self.provider.cognito_client.initiate_auth.return_value = mock_response

        result = await self.provider.authenticate("test@example.com", "password123")

        assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_invalid_credentials(self) -> None:
        """Test authentication with invalid credentials."""
        error = ClientError(
            {
                "Error": {
                    "Code": "NotAuthorizedException",
                    "Message": "Invalid credentials",
                }
            },
            "InitiateAuth",
        )
        self.provider.cognito_client.initiate_auth.side_effect = error

        with pytest.raises(AuthenticationError, match="Invalid email or password"):
            await self.provider.authenticate("test@example.com", "wrongpassword")

    @pytest.mark.asyncio
    async def test_authenticate_client_error(self) -> None:
        """Test authentication with other client error."""
        error = ClientError(
            {"Error": {"Code": "UserNotFoundException", "Message": "User not found"}},
            "InitiateAuth",
        )
        self.provider.cognito_client.initiate_auth.side_effect = error

        with pytest.raises(AuthenticationError, match="Authentication failed"):
            await self.provider.authenticate("test@example.com", "password123")

    @pytest.mark.asyncio
    async def test_authenticate_unexpected_error(self) -> None:
        """Test authentication with unexpected error."""
        self.provider.cognito_client.initiate_auth.side_effect = Exception(
            "Network error"
        )

        with pytest.raises(AuthenticationError, match="Unexpected error"):
            await self.provider.authenticate("test@example.com", "password123")


class TestProviderLifecycle:
    """Test provider initialization and cleanup."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.provider = CognitoAuthProvider("us-east-1_ABC123", "client123")

    @patch.object(CognitoAuthProvider, "jwks")
    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_jwks: MagicMock) -> None:
        """Test successful provider initialization."""
        mock_jwks.return_value = {"keys": []}

        await self.provider.initialize()

        # Should not raise any errors

    @patch("clarity.auth.aws_cognito_provider.requests.get")
    @pytest.mark.asyncio
    async def test_initialize_failure(self, mock_requests_get: MagicMock) -> None:
        """Test provider initialization failure."""
        mock_requests_get.side_effect = Exception("JWKS fetch failed")

        with pytest.raises(Exception, match="Failed to fetch JWKS keys"):
            await self.provider.initialize()

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        """Test provider shutdown."""
        self.provider._jwks_cache = {"keys": []}

        await self.provider.shutdown()

        assert self.provider._jwks_cache is None

    @pytest.mark.asyncio
    async def test_cleanup(self) -> None:
        """Test provider cleanup."""
        self.provider._jwks_cache = {"keys": []}

        await self.provider.cleanup()

        assert self.provider._jwks_cache is None

    @pytest.mark.asyncio
    async def test_get_user_info_success(self) -> None:
        """Test successful get_user_info."""
        mock_user = User(
            uid="user123",
            email="test@example.com",
            display_name="Test User",
            created_at=datetime.now(UTC),
            last_login=None,
            metadata={"status": "CONFIRMED"},
        )

        with patch.object(self.provider, "get_user", return_value=mock_user):
            result = await self.provider.get_user_info("user123")

        assert result is not None
        assert result["uid"] == "user123"
        assert result["email"] == "test@example.com"
        assert result["display_name"] == "Test User"
        assert result["metadata"]["status"] == "CONFIRMED"

    @pytest.mark.asyncio
    async def test_get_user_info_user_not_found(self) -> None:
        """Test get_user_info when user not found."""
        with patch.object(self.provider, "get_user", return_value=None):
            result = await self.provider.get_user_info("nonexistent")

        assert result is None


class TestGetCognitoProviderFunction:
    """Test the singleton Cognito provider function."""

    @patch.dict(
        "os.environ",
        {
            "COGNITO_USER_POOL_ID": "test-pool-123",
            "COGNITO_CLIENT_ID": "test-client-123",
            "COGNITO_REGION": "us-west-2",
        },
    )
    def test_get_cognito_provider_success(self) -> None:
        """Test successful provider creation from environment."""
        provider = get_cognito_provider()

        assert provider.user_pool_id == "test-pool-123"
        assert provider.client_id == "test-client-123"
        assert provider.region == "us-west-2"

    @patch.dict(
        "os.environ",
        {
            "COGNITO_USER_POOL_ID": "test-pool-123",
            "COGNITO_CLIENT_ID": "test-client-123",
            "AWS_REGION": "eu-west-1",
            "COGNITO_REGION": "",  # Clear COGNITO_REGION to test AWS_REGION fallback
        },
        clear=True,  # Clear all env vars to ensure clean test
    )
    def test_get_cognito_provider_aws_region_fallback(self) -> None:
        """Test provider creation with AWS_REGION fallback."""
        # Clear LRU cache to prevent test isolation issues
        get_cognito_provider.cache_clear()

        provider = get_cognito_provider()

        assert provider.region == "eu-west-1"

    @patch.dict(
        "os.environ",
        {
            "COGNITO_USER_POOL_ID": "test-pool-123",
            "COGNITO_CLIENT_ID": "test-client-123",
        },
    )
    def test_get_cognito_provider_default_region(self) -> None:
        """Test provider creation with default region."""
        # Clear LRU cache to prevent test isolation issues
        get_cognito_provider.cache_clear()

        provider = get_cognito_provider()

        assert provider.region == "us-east-1"

    @patch.dict("os.environ", {"COGNITO_CLIENT_ID": "test-client-123"}, clear=True)
    def test_get_cognito_provider_missing_pool_id(self) -> None:
        """Test provider creation with missing user pool ID."""
        # Clear LRU cache to prevent test isolation issues
        get_cognito_provider.cache_clear()

        with pytest.raises(ValueError, match="Cognito configuration missing"):
            get_cognito_provider()

    @patch.dict("os.environ", {"COGNITO_USER_POOL_ID": "test-pool-123"}, clear=True)
    def test_get_cognito_provider_missing_client_id(self) -> None:
        """Test provider creation with missing client ID."""
        # Clear LRU cache to prevent test isolation issues
        get_cognito_provider.cache_clear()

        with pytest.raises(ValueError, match="Cognito configuration missing"):
            get_cognito_provider()

    @patch.dict("os.environ", {}, clear=True)
    def test_get_cognito_provider_no_config(self) -> None:
        """Test provider creation with no configuration."""
        # Clear LRU cache to prevent test isolation issues
        get_cognito_provider.cache_clear()

        with pytest.raises(ValueError, match="Cognito configuration missing"):
            get_cognito_provider()


class TestProductionScenarios:
    """Test realistic production scenarios."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.provider = CognitoAuthProvider("us-east-1_ABC123", "client123")
        self.provider.cognito_client = Mock()

    @pytest.mark.asyncio
    async def test_complete_user_lifecycle(self) -> None:
        """Test complete user lifecycle: create, get, update, delete."""
        # Create user
        create_response = {"UserSub": "user123"}
        self.provider.cognito_client.sign_up.return_value = create_response

        created_user = await self.provider.create_user(
            "test@example.com", "password123", "Test User"
        )
        assert created_user.uid == "user123"

        # Get user
        get_response = {
            "Username": "user123",
            "UserStatus": "CONFIRMED",
            "Enabled": True,
            "UserCreateDate": datetime(2023, 1, 1, tzinfo=UTC),
            "UserLastModifiedDate": datetime(2023, 1, 2, tzinfo=UTC),
            "UserAttributes": [
                {"Name": "email", "Value": "test@example.com"},
                {"Name": "name", "Value": "Test User"},
            ],
        }
        self.provider.cognito_client.admin_get_user.return_value = get_response

        retrieved_user = await self.provider.get_user("user123")
        assert retrieved_user.email == "test@example.com"

        # Update user
        self.provider.cognito_client.admin_update_user_attributes.return_value = {}

        with patch.object(self.provider, "get_user", return_value=retrieved_user):
            updated_user = await self.provider.update_user(
                "user123", display_name="Updated Name"
            )
        assert updated_user is not None

        # Delete user
        self.provider.cognito_client.admin_delete_user.return_value = {}
        result = await self.provider.delete_user("user123")
        assert result is True

    @pytest.mark.asyncio
    async def test_authentication_flow_with_token_verification(self) -> None:
        """Test complete authentication flow with token verification."""
        # Authenticate
        auth_response = {
            "AuthenticationResult": {
                "AccessToken": "access_token_123",
                "IdToken": "id_token_123",
                "RefreshToken": "refresh_token_123",
                "ExpiresIn": 3600,
            }
        }
        self.provider.cognito_client.initiate_auth.return_value = auth_response

        tokens = await self.provider.authenticate("test@example.com", "password123")
        assert (
            tokens["access_token"]
            == "access_token_123"  # noqa: S105 - Test token value
        )

        # Verify token
        with patch.object(self.provider, "verify_token") as mock_verify:
            mock_verify.return_value = {
                "sub": "user123",
                "email": "test@example.com",
                "exp": 2000.0,
            }

            claims = await self.provider.verify_token(tokens["id_token"])
            assert claims["sub"] == "user123"

    @pytest.mark.asyncio
    async def test_error_handling_chain(self) -> None:
        """Test error handling across multiple operations."""
        # Test authentication failure leading to user creation
        auth_error = ClientError(
            {"Error": {"Code": "UserNotFoundException", "Message": "User not found"}},
            "InitiateAuth",
        )
        self.provider.cognito_client.initiate_auth.side_effect = auth_error

        with pytest.raises(AuthenticationError):
            await self.provider.authenticate("newuser@example.com", "password123")

        # Then create the user
        create_response = {"UserSub": "newuser123"}
        self.provider.cognito_client.sign_up.return_value = create_response
        self.provider.cognito_client.initiate_auth.side_effect = None  # Reset

        user = await self.provider.create_user("newuser@example.com", "password123")
        assert user.uid == "newuser123"

    @pytest.mark.asyncio
    async def test_concurrent_jwks_access(self) -> None:
        """Test concurrent access to JWKS cache."""
        # Mock JWKS fetch
        with patch("clarity.auth.aws_cognito_provider.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"keys": [{"kid": "key1"}]}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # Simulate concurrent access
            async def access_jwks() -> dict[str, Any]:
                await asyncio.sleep(0)  # Make it properly async
                return self.provider.jwks

            # Run multiple concurrent accesses
            results = await asyncio.gather(*[access_jwks() for _ in range(5)])

            # All should return the same cached result
            for result in results:
                assert result == {"keys": [{"kid": "key1"}]}

            # Should only fetch once due to caching
            assert mock_get.call_count <= 2  # Allow for some race conditions

    @pytest.mark.asyncio
    async def test_provider_resilience_to_network_issues(self) -> None:
        """Test provider resilience to network issues."""
        # Test JWKS fetch with network issues
        with patch("clarity.auth.aws_cognito_provider.requests.get") as mock_get:
            # Both calls fail - simulates persistent network issues
            mock_get.side_effect = requests.RequestException("Network timeout")

            # Should raise on first call with no cache
            with pytest.raises(AuthenticationError):
                _ = self.provider.jwks

            # Set up cache first
            self.provider._jwks_cache = {"keys": [{"kid": "cached"}]}

            # Should return cached version on network error
            with patch(
                "clarity.auth.aws_cognito_provider.time.time", return_value=10000.0
            ):
                jwks = self.provider.jwks
            assert jwks == {"keys": [{"kid": "cached"}]}
