"""Comprehensive tests for Cognito authentication service."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch
import uuid

from botocore.exceptions import ClientError
import pytest

from clarity.models.auth import (
    LoginResponse,
    RegistrationResponse,
    TokenResponse,
    UserLoginRequest,
    UserRegistrationRequest,
    UserRole,
    UserSessionResponse,
    UserStatus,
)
from clarity.ports.auth_ports import IAuthProvider
from clarity.services.cognito_auth_service import (
    AccountDisabledError,
    AuthenticationError,
    CognitoAuthenticationService,
    EmailNotVerifiedError,
    InvalidCredentialsError,
    UserAlreadyExistsError,
    UserNotFoundError,
)
from clarity.services.dynamodb_service import DynamoDBService


@pytest.fixture
def mock_auth_provider():
    """Mock authentication provider."""
    return Mock(spec=IAuthProvider)


@pytest.fixture
def mock_dynamodb_service():
    """Mock DynamoDB service."""
    mock = Mock(spec=DynamoDBService)
    mock.put_item = AsyncMock(return_value="test-id")
    mock.get_item = AsyncMock(return_value=None)
    mock.update_item = AsyncMock(return_value=True)
    mock.query = AsyncMock(return_value={"Items": []})
    return mock


@pytest.fixture
def mock_cognito_client():
    """Mock Cognito client."""
    with patch("boto3.client") as mock_client:
        yield mock_client.return_value


@pytest.fixture
def cognito_service(mock_auth_provider, mock_dynamodb_service, mock_cognito_client):
    """Create Cognito authentication service with mocked dependencies."""
    service = CognitoAuthenticationService(
        auth_provider=mock_auth_provider,
        dynamodb_service=mock_dynamodb_service,
        user_pool_id="test-pool-id",
        client_id="test-client-id",
        client_secret="test-client-secret",
        region="us-east-1",
    )
    service.cognito_client = mock_cognito_client
    return service


@pytest.fixture
def cognito_service_no_secret(
    mock_auth_provider, mock_dynamodb_service, mock_cognito_client
):
    """Create Cognito authentication service without client secret."""
    service = CognitoAuthenticationService(
        auth_provider=mock_auth_provider,
        dynamodb_service=mock_dynamodb_service,
        user_pool_id="test-pool-id",
        client_id="test-client-id",
        client_secret=None,
        region="us-east-1",
    )
    service.cognito_client = mock_cognito_client
    return service


@pytest.fixture
def valid_registration_request():
    """Create valid registration request."""
    return UserRegistrationRequest(
        email="test@example.com",
        password="SecurePassword123!",
        first_name="John",
        last_name="Doe",
        phone_number="+1234567890",
        terms_accepted=True,
        privacy_policy_accepted=True,
    )


@pytest.fixture
def valid_login_request():
    """Create valid login request."""
    return UserLoginRequest(
        email="test@example.com",
        password="SecurePassword123!",
        remember_me=False,
    )


class TestCognitoAuthServiceInit:
    """Test Cognito authentication service initialization."""

    def test_init_with_default_params(self, mock_auth_provider, mock_dynamodb_service):
        """Test initialization with default parameters."""
        with patch("boto3.client") as mock_boto:
            service = CognitoAuthenticationService(
                auth_provider=mock_auth_provider,
                dynamodb_service=mock_dynamodb_service,
                user_pool_id="test-pool",
                client_id="test-client",
            )

            assert service.user_pool_id == "test-pool"
            assert service.client_id == "test-client"
            assert service.client_secret is None
            assert service.region == "us-east-1"
            assert service.default_token_expiry == 3600
            assert service.refresh_token_expiry == 86400 * 30

            mock_boto.assert_called_once_with("cognito-idp", region_name="us-east-1")

    def test_init_with_custom_params(self, mock_auth_provider, mock_dynamodb_service):
        """Test initialization with custom parameters."""
        with patch("boto3.client"):
            service = CognitoAuthenticationService(
                auth_provider=mock_auth_provider,
                dynamodb_service=mock_dynamodb_service,
                user_pool_id="custom-pool",
                client_id="custom-client",
                client_secret="custom-secret",
                region="eu-west-1",
                default_token_expiry=7200,
                refresh_token_expiry=86400 * 60,
            )

            assert service.user_pool_id == "custom-pool"
            assert service.client_id == "custom-client"
            assert service.client_secret == "custom-secret"
            assert service.region == "eu-west-1"
            assert service.default_token_expiry == 7200
            assert service.refresh_token_expiry == 86400 * 60

    def test_table_names(self, cognito_service):
        """Test table name configuration."""
        assert cognito_service.users_table == "clarity_users"
        assert cognito_service.sessions_table == "clarity_user_sessions"
        assert cognito_service.refresh_tokens_table == "clarity_refresh_tokens"


class TestComputeSecretHash:
    """Test secret hash computation."""

    def test_compute_secret_hash_with_secret(self, cognito_service):
        """Test secret hash computation with client secret."""
        username = "test@example.com"
        secret_hash = cognito_service._compute_secret_hash(username)

        assert secret_hash is not None
        assert isinstance(secret_hash, str)
        # Verify it's base64 encoded
        assert len(secret_hash) > 0

    def test_compute_secret_hash_without_secret(self, cognito_service_no_secret):
        """Test secret hash computation without client secret."""
        username = "test@example.com"
        secret_hash = cognito_service_no_secret._compute_secret_hash(username)

        assert secret_hash is None


class TestRegisterUser:
    """Test user registration functionality."""

    @pytest.mark.asyncio
    async def test_register_user_success(
        self,
        cognito_service,
        mock_cognito_client,
        mock_dynamodb_service,
        valid_registration_request,
    ):
        """Test successful user registration."""
        # Mock Cognito response
        user_sub = str(uuid.uuid4())
        mock_cognito_client.sign_up.return_value = {
            "UserSub": user_sub,
            "CodeDeliveryDetails": {
                "Destination": "test@example.com",
                "DeliveryMedium": "EMAIL",
            },
        }

        # Register user
        response = await cognito_service.register_user(
            valid_registration_request,
            device_info={"device_id": "test-device"},
        )

        # Verify response
        assert isinstance(response, RegistrationResponse)
        assert response.user_id == uuid.UUID(user_sub)
        assert response.email == "test@example.com"
        assert response.status == UserStatus.PENDING_VERIFICATION
        assert response.verification_email_sent is True

        # Verify Cognito was called
        mock_cognito_client.sign_up.assert_called_once()
        call_args = mock_cognito_client.sign_up.call_args[1]
        assert call_args["ClientId"] == "test-client-id"
        assert call_args["Username"] == "test@example.com"
        assert call_args["Password"] == "SecurePassword123!"
        assert "SecretHash" in call_args

        # Verify user attributes
        user_attrs = {
            attr["Name"]: attr["Value"] for attr in call_args["UserAttributes"]
        }
        assert user_attrs["email"] == "test@example.com"
        assert user_attrs["given_name"] == "John"
        assert user_attrs["family_name"] == "Doe"
        assert user_attrs["phone_number"] == "+1234567890"

        # Verify DynamoDB was called
        mock_dynamodb_service.put_item.assert_called_once()
        db_call_args = mock_dynamodb_service.put_item.call_args
        assert db_call_args[1]["table_name"] == "clarity_users"

        item = db_call_args[1]["item"]
        assert item["user_id"] == user_sub
        assert item["email"] == "test@example.com"
        assert item["first_name"] == "John"
        assert item["last_name"] == "Doe"
        assert item["status"] == UserStatus.PENDING_VERIFICATION.value
        assert item["role"] == UserRole.PATIENT.value
        assert item["device_info"] == {"device_id": "test-device"}

    @pytest.mark.asyncio
    async def test_register_user_no_phone(
        self, cognito_service, mock_cognito_client, valid_registration_request
    ):
        """Test user registration without phone number."""
        valid_registration_request.phone_number = None
        user_sub = str(uuid.uuid4())
        mock_cognito_client.sign_up.return_value = {"UserSub": user_sub}

        await cognito_service.register_user(valid_registration_request)

        # Verify phone number not in attributes
        call_args = mock_cognito_client.sign_up.call_args[1]
        user_attrs = {
            attr["Name"]: attr["Value"] for attr in call_args["UserAttributes"]
        }
        assert "phone_number" not in user_attrs

    @pytest.mark.asyncio
    async def test_register_user_already_exists(
        self, cognito_service, mock_cognito_client, valid_registration_request
    ):
        """Test registration when user already exists."""
        mock_cognito_client.sign_up.side_effect = ClientError(
            {"Error": {"Code": "UsernameExistsException", "Message": "User exists"}},
            "SignUp",
        )

        with pytest.raises(UserAlreadyExistsError) as exc_info:
            await cognito_service.register_user(valid_registration_request)

        assert "already exists" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_register_user_cognito_error(
        self, cognito_service, mock_cognito_client, valid_registration_request
    ):
        """Test registration with Cognito error."""
        mock_cognito_client.sign_up.side_effect = ClientError(
            {
                "Error": {
                    "Code": "InvalidParameterException",
                    "Message": "Invalid param",
                }
            },
            "SignUp",
        )

        with pytest.raises(AuthenticationError) as exc_info:
            await cognito_service.register_user(valid_registration_request)

        assert "Registration failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_register_user_unexpected_error(
        self, cognito_service, mock_cognito_client, valid_registration_request
    ):
        """Test registration with unexpected error."""
        mock_cognito_client.sign_up.side_effect = Exception("Unexpected error")

        with pytest.raises(AuthenticationError) as exc_info:
            await cognito_service.register_user(valid_registration_request)

        assert "Registration failed" in str(exc_info.value)


class TestLoginUser:
    """Test user login functionality."""

    @pytest.mark.asyncio
    async def test_login_user_success(
        self,
        cognito_service,
        mock_cognito_client,
        mock_dynamodb_service,
        valid_login_request,
    ):
        """Test successful user login."""
        # Mock Cognito responses
        user_sub = str(uuid.uuid4())
        mock_cognito_client.admin_initiate_auth.return_value = {
            "AuthenticationResult": {
                "AccessToken": "test-access-token",
                "RefreshToken": "test-refresh-token",
                "TokenType": "Bearer",
            }
        }
        mock_cognito_client.get_user.return_value = {
            "UserAttributes": [
                {"Name": "sub", "Value": user_sub},
                {"Name": "email", "Value": "test@example.com"},
            ]
        }

        # Mock DynamoDB user data
        mock_dynamodb_service.get_item.return_value = {
            "user_id": user_sub,
            "email": "test@example.com",
            "first_name": "John",
            "last_name": "Doe",
            "role": UserRole.PATIENT.value,
            "permissions": ["read", "write"],
            "status": UserStatus.ACTIVE.value,
            "mfa_enabled": False,
            "email_verified": True,
            "created_at": datetime.now(UTC).isoformat(),
            "last_login": datetime.now(UTC).isoformat(),
        }

        # Login user
        response = await cognito_service.login_user(
            valid_login_request,
            device_info={"device_id": "test-device"},
            ip_address="192.168.1.1",
        )

        # Verify response
        assert isinstance(response, LoginResponse)
        assert response.user.email == "test@example.com"
        assert response.user.first_name == "John"
        assert response.user.last_name == "Doe"
        assert response.user.role == UserRole.PATIENT.value
        assert response.tokens.access_token == "test-access-token"
        assert response.tokens.refresh_token == "test-refresh-token"
        assert response.requires_mfa is False

        # Verify Cognito calls
        mock_cognito_client.admin_initiate_auth.assert_called_once()
        auth_call = mock_cognito_client.admin_initiate_auth.call_args[1]
        assert auth_call["UserPoolId"] == "test-pool-id"
        assert auth_call["ClientId"] == "test-client-id"
        assert auth_call["AuthFlow"] == "ADMIN_NO_SRP_AUTH"
        assert auth_call["AuthParameters"]["USERNAME"] == "test@example.com"
        assert auth_call["AuthParameters"]["PASSWORD"] == "SecurePassword123!"
        assert "SECRET_HASH" in auth_call["AuthParameters"]

        # Verify DynamoDB update
        mock_dynamodb_service.update_item.assert_called_once()
        update_call = mock_dynamodb_service.update_item.call_args[1]
        assert update_call["table_name"] == "clarity_users"
        assert update_call["key"]["user_id"] == user_sub

    @pytest.mark.asyncio
    async def test_login_user_no_db_data(
        self,
        cognito_service,
        mock_cognito_client,
        mock_dynamodb_service,
        valid_login_request,
    ):
        """Test login when user data not in database."""
        # Mock Cognito responses
        user_sub = str(uuid.uuid4())
        mock_cognito_client.admin_initiate_auth.return_value = {
            "AuthenticationResult": {
                "AccessToken": "test-access-token",
                "RefreshToken": "test-refresh-token",
            }
        }
        mock_cognito_client.get_user.return_value = {
            "UserAttributes": [{"Name": "sub", "Value": user_sub}]
        }

        # No user data in DB
        mock_dynamodb_service.get_item.return_value = None

        # Login should still succeed
        response = await cognito_service.login_user(valid_login_request)

        assert response.user.email == "test@example.com"
        assert response.user.first_name == "test"  # Default to email prefix
        assert response.user.last_name == ""
        assert response.user.role == UserRole.PATIENT.value

    @pytest.mark.asyncio
    async def test_login_user_password_change_required(
        self, cognito_service, mock_cognito_client, valid_login_request
    ):
        """Test login when password change is required."""
        mock_cognito_client.admin_initiate_auth.return_value = {
            "ChallengeName": "NEW_PASSWORD_REQUIRED",
            "Session": "test-session",
        }

        with pytest.raises(AuthenticationError) as exc_info:
            await cognito_service.login_user(valid_login_request)

        assert "Password change required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_login_user_mfa_setup(
        self, cognito_service, mock_cognito_client, valid_login_request
    ):
        """Test login when MFA setup is required."""
        mock_cognito_client.admin_initiate_auth.return_value = {
            "ChallengeName": "MFA_SETUP",
            "Session": "test-session",
        }

        response = await cognito_service.login_user(valid_login_request)

        assert response.requires_mfa is True
        assert response.mfa_session_token is not None
        assert response.tokens.access_token == ""

    @pytest.mark.asyncio
    async def test_login_user_not_found(
        self, cognito_service, mock_cognito_client, valid_login_request
    ):
        """Test login when user not found."""
        mock_cognito_client.admin_initiate_auth.side_effect = ClientError(
            {"Error": {"Code": "UserNotFoundException", "Message": "User not found"}},
            "AdminInitiateAuth",
        )

        with pytest.raises(UserNotFoundError) as exc_info:
            await cognito_service.login_user(valid_login_request)

        assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_login_user_invalid_credentials(
        self, cognito_service, mock_cognito_client, valid_login_request
    ):
        """Test login with invalid credentials."""
        mock_cognito_client.admin_initiate_auth.side_effect = ClientError(
            {
                "Error": {
                    "Code": "NotAuthorizedException",
                    "Message": "Invalid credentials",
                }
            },
            "AdminInitiateAuth",
        )

        with pytest.raises(InvalidCredentialsError) as exc_info:
            await cognito_service.login_user(valid_login_request)

        assert "Invalid username or password" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_login_user_email_not_verified(
        self, cognito_service, mock_cognito_client, valid_login_request
    ):
        """Test login when email not verified."""
        mock_cognito_client.admin_initiate_auth.side_effect = ClientError(
            {
                "Error": {
                    "Code": "UserNotConfirmedException",
                    "Message": "User not confirmed",
                }
            },
            "AdminInitiateAuth",
        )

        with pytest.raises(EmailNotVerifiedError) as exc_info:
            await cognito_service.login_user(valid_login_request)

        assert "Email verification required" in str(exc_info.value)


class TestRefreshAccessToken:
    """Test token refresh functionality."""

    @pytest.mark.asyncio
    async def test_refresh_access_token_success(
        self, cognito_service, mock_cognito_client
    ):
        """Test successful token refresh."""
        mock_cognito_client.admin_initiate_auth.return_value = {
            "AuthenticationResult": {
                "AccessToken": "new-access-token",
                "TokenType": "Bearer",
            }
        }

        response = await cognito_service.refresh_access_token("test-refresh-token")

        assert isinstance(response, TokenResponse)
        assert response.access_token == "new-access-token"
        assert response.refresh_token == "test-refresh-token"
        assert response.token_type == "bearer"

        # Verify Cognito call
        mock_cognito_client.admin_initiate_auth.assert_called_once()
        call_args = mock_cognito_client.admin_initiate_auth.call_args[1]
        assert call_args["AuthFlow"] == "REFRESH_TOKEN_AUTH"
        assert call_args["AuthParameters"]["REFRESH_TOKEN"] == "test-refresh-token"

    @pytest.mark.asyncio
    async def test_refresh_access_token_invalid(
        self, cognito_service, mock_cognito_client
    ):
        """Test token refresh with invalid token."""
        mock_cognito_client.admin_initiate_auth.side_effect = ClientError(
            {"Error": {"Code": "NotAuthorizedException", "Message": "Invalid token"}},
            "AdminInitiateAuth",
        )

        with pytest.raises(InvalidCredentialsError) as exc_info:
            await cognito_service.refresh_access_token("invalid-token")

        assert "Invalid refresh token" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_refresh_access_token_error(
        self, cognito_service, mock_cognito_client
    ):
        """Test token refresh with error."""
        mock_cognito_client.admin_initiate_auth.side_effect = ClientError(
            {"Error": {"Code": "InternalError", "Message": "Internal error"}},
            "AdminInitiateAuth",
        )

        with pytest.raises(AuthenticationError) as exc_info:
            await cognito_service.refresh_access_token("test-token")

        assert "Token refresh failed" in str(exc_info.value)


class TestLogoutUser:
    """Test user logout functionality."""

    @pytest.mark.asyncio
    async def test_logout_user_success(
        self, cognito_service, mock_cognito_client, mock_dynamodb_service
    ):
        """Test successful logout."""
        # Mock DynamoDB query response
        mock_dynamodb_service.query.return_value = {
            "Items": [
                {"session_id": "session-1", "refresh_token": "test-refresh-token"},
                {"session_id": "session-2", "refresh_token": "test-refresh-token"},
            ]
        }

        result = await cognito_service.logout_user("test-refresh-token")

        assert result is True

        # Verify token revocation
        mock_cognito_client.revoke_token.assert_called_once_with(
            Token="test-refresh-token",
            ClientId="test-client-id",
        )

        # Verify sessions were deactivated
        assert mock_dynamodb_service.update_item.call_count == 2

    @pytest.mark.asyncio
    async def test_logout_user_error(self, cognito_service, mock_cognito_client):
        """Test logout with error."""
        mock_cognito_client.revoke_token.side_effect = Exception("Revoke error")

        result = await cognito_service.logout_user("test-refresh-token")

        assert result is False


class TestGetUserById:
    """Test get user by ID functionality."""

    @pytest.mark.asyncio
    async def test_get_user_by_id_success(self, cognito_service, mock_dynamodb_service):
        """Test successful user retrieval."""
        user_id = str(uuid.uuid4())
        mock_dynamodb_service.get_item.return_value = {
            "user_id": user_id,
            "email": "test@example.com",
            "first_name": "John",
            "last_name": "Doe",
            "role": UserRole.PATIENT.value,
            "permissions": [],
            "status": UserStatus.ACTIVE.value,
            "mfa_enabled": False,
            "email_verified": True,
            "created_at": datetime.now(UTC).isoformat(),
        }

        user = await cognito_service.get_user_by_id(user_id)

        assert user is not None
        assert user.user_id == uuid.UUID(user_id)
        assert user.email == "test@example.com"
        assert user.first_name == "John"

    @pytest.mark.asyncio
    async def test_get_user_by_id_not_found(
        self, cognito_service, mock_dynamodb_service
    ):
        """Test user retrieval when not found."""
        mock_dynamodb_service.get_item.return_value = None

        user = await cognito_service.get_user_by_id("non-existent")

        assert user is None

    @pytest.mark.asyncio
    async def test_get_user_by_id_error(self, cognito_service, mock_dynamodb_service):
        """Test user retrieval with error."""
        mock_dynamodb_service.get_item.side_effect = Exception("DB error")

        user = await cognito_service.get_user_by_id("test-id")

        assert user is None


class TestVerifyEmail:
    """Test email verification functionality."""

    @pytest.mark.asyncio
    async def test_verify_email(self, cognito_service):
        """Test email verification."""
        result = await cognito_service.verify_email("test-code")
        assert result is True


class TestHelperMethods:
    """Test helper methods."""

    @pytest.mark.asyncio
    async def test_create_user_session(self, cognito_service, mock_dynamodb_service):
        """Test user session creation."""
        user_id = str(uuid.uuid4())

        session_id = await cognito_service._create_user_session(
            user_id=user_id,
            refresh_token="test-token",
            device_info={"device": "test"},
            ip_address="192.168.1.1",
            remember_me=True,
        )

        assert session_id is not None
        mock_dynamodb_service.put_item.assert_called_once()

        call_args = mock_dynamodb_service.put_item.call_args[1]
        assert call_args["table_name"] == "clarity_user_sessions"

        session_data = call_args["item"]
        assert session_data["user_id"] == user_id
        assert session_data["refresh_token"] == "test-token"
        assert session_data["device_info"] == {"device": "test"}
        assert session_data["ip_address"] == "192.168.1.1"
        assert session_data["is_active"] is True

    @pytest.mark.asyncio
    async def test_create_user_session_response(self, cognito_service):
        """Test user session response creation."""
        user_id = str(uuid.uuid4())
        user_data = {
            "email": "test@example.com",
            "first_name": "John",
            "last_name": "Doe",
            "role": UserRole.PATIENT.value,
            "permissions": ["read"],
            "status": UserStatus.ACTIVE.value,
            "mfa_enabled": True,
            "email_verified": True,
            "created_at": datetime.now(UTC).isoformat(),
            "last_login": datetime.now(UTC).isoformat(),
        }

        response = await cognito_service._create_user_session_response(
            user_id, user_data
        )

        assert isinstance(response, UserSessionResponse)
        assert response.user_id == uuid.UUID(user_id)
        assert response.email == "test@example.com"
        assert response.first_name == "John"
        assert response.mfa_enabled is True

    def test_handle_mfa_setup(self, cognito_service, valid_login_request):
        """Test MFA setup handling."""
        auth_response = {
            "ChallengeName": "MFA_SETUP",
            "Session": "test-session",
        }

        response = cognito_service._handle_mfa_setup(
            auth_response,
            valid_login_request,
            device_info=None,
            ip_address=None,
        )

        assert isinstance(response, LoginResponse)
        assert response.requires_mfa is True
        assert response.mfa_session_token is not None
        assert response.tokens.scope == "mfa_pending"


class TestExceptionClasses:
    """Test custom exception classes."""

    def test_authentication_error(self):
        """Test AuthenticationError exception."""
        error = AuthenticationError("Test error")
        assert str(error) == "Test error"

    def test_user_not_found_error(self):
        """Test UserNotFoundError exception."""
        error = UserNotFoundError("User not found")
        assert str(error) == "User not found"
        assert isinstance(error, AuthenticationError)

    def test_invalid_credentials_error(self):
        """Test InvalidCredentialsError exception."""
        error = InvalidCredentialsError("Invalid credentials")
        assert str(error) == "Invalid credentials"
        assert isinstance(error, AuthenticationError)

    def test_user_already_exists_error(self):
        """Test UserAlreadyExistsError exception."""
        error = UserAlreadyExistsError("User exists")
        assert str(error) == "User exists"
        assert isinstance(error, AuthenticationError)

    def test_email_not_verified_error(self):
        """Test EmailNotVerifiedError exception."""
        error = EmailNotVerifiedError("Email not verified")
        assert str(error) == "Email not verified"
        assert isinstance(error, AuthenticationError)

    def test_account_disabled_error(self):
        """Test AccountDisabledError exception."""
        error = AccountDisabledError("Account disabled")
        assert str(error) == "Account disabled"
        assert isinstance(error, AuthenticationError)
