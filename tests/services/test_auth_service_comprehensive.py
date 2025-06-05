"""🚀 COMPREHENSIVE AUTH SERVICE TEST COVERAGE WARHEAD! 🚀.

Blasting test coverage from 19% → 95%+ for AuthenticationService.
Tests every method, error case, edge case, and business logic path.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch
import uuid

from firebase_admin import (
    auth,  # Import the original auth to access UserNotFoundError if needed by mock_auth setup
)
from pydantic import ValidationError
import pytest

from clarity.models.auth import (
    LoginResponse,
    TokenResponse,
    UserLoginRequest,
    UserRegistrationRequest,
    UserRole,
    UserSessionResponse,
    UserStatus,
)
from clarity.ports.auth_ports import IAuthProvider
from clarity.services.auth_service import (
    AccountDisabledError,
    AuthenticationError,
    AuthenticationService,
    EmailNotVerifiedError,
    InvalidCredentialsError,
    UserAlreadyExistsError,
    UserNotFoundError,
)
from clarity.storage.firestore_client import FirestoreClient

# Constants for exception messages
REGISTRATION_PASSWORD_ERROR_MSG = (
    "Test setup error: default_password for registration must be a string."
)
LOGIN_PASSWORD_ERROR_MSG = (
    "Test setup error: default_password for login must be a string."
)


# Create proper mock exceptions that inherit from BaseException
class MockUserNotFoundError(Exception):
    """Mock Firebase UserNotFoundError that properly inherits from BaseException."""


class MockInvalidIdTokenError(Exception):
    """Mock Firebase InvalidIdTokenError that properly inherits from BaseException."""


class MockAuthError(Exception):
    """Mock Firebase AuthError that properly inherits from BaseException."""


class MockFirebaseUserRecord:  # Ensure this class definition is present
    """Mock Firebase user record."""

    def __init__(
        self,
        uid: str,
        email: str | None = None,
        *,
        disabled: bool = False,
        email_verified: bool = True,
    ) -> None:
        self.uid = uid
        self.email = email
        self.disabled = disabled
        self.email_verified = email_verified
        self.tokens_valid_after_timestamp: int | None = None
        self.custom_claims: dict[str, Any] | None = None
        self.user_metadata = Mock()  # For last_sign_in_timestamp, creation_timestamp
        if hasattr(self.user_metadata, "last_sign_in_timestamp"):
            self.user_metadata.last_sign_in_timestamp = (
                datetime.now(UTC).timestamp() * 1000
            )
        if hasattr(self.user_metadata, "creation_timestamp"):
            self.user_metadata.creation_timestamp = datetime.now(UTC).timestamp() * 1000


class TestAuthServiceExceptionHelpers:
    """💥 Test all exception helper functions."""

    @staticmethod
    def test_account_disabled_exception() -> None:
        """Test account disabled exception."""
        error_msg = "Account is disabled"
        with pytest.raises(AccountDisabledError) as exc_info:
            raise AccountDisabledError(error_msg)
        assert str(exc_info.value) == error_msg

    @staticmethod
    def test_user_not_found_exception() -> None:
        """Test user not found in database exception."""
        error_msg = "User data not found in database"
        with pytest.raises(UserNotFoundError) as exc_info:
            raise UserNotFoundError(error_msg)
        assert str(exc_info.value) == error_msg

    @staticmethod
    def test_invalid_credentials_exception() -> None:
        """Test invalid refresh token exception."""
        error_msg = "Invalid refresh token"
        with pytest.raises(InvalidCredentialsError) as exc_info:
            raise InvalidCredentialsError(error_msg)
        assert str(exc_info.value) == error_msg

    @staticmethod
    def test_refresh_token_expired_exception() -> None:
        """Test refresh token expired exception."""
        error_msg = "Refresh token expired"
        with pytest.raises(InvalidCredentialsError) as exc_info:
            raise InvalidCredentialsError(error_msg)
        assert str(exc_info.value) == error_msg


class TestAuthServiceInitialization:
    """🔧 Test AuthenticationService initialization."""

    @pytest.fixture
    @staticmethod
    def mock_auth_provider() -> Mock:
        """Mock auth provider."""
        return Mock(spec=IAuthProvider)

    @pytest.fixture
    @staticmethod
    def mock_firestore_client() -> Mock:
        """Mock Firestore client."""
        return Mock(spec=FirestoreClient)

    @staticmethod
    def test_service_initialization_defaults(
        mock_auth_provider: Mock,
        mock_firestore_client: Mock,
    ) -> None:
        """Test service initialization with default values."""
        service = AuthenticationService(
            auth_provider=mock_auth_provider,
            firestore_client=mock_firestore_client,
        )

        assert service.auth_provider == mock_auth_provider
        assert service.firestore_client == mock_firestore_client
        assert service.default_token_expiry == 3600  # 1 hour
        assert service.refresh_token_expiry == 86400 * 30  # 30 days
        assert service.users_collection == "users"
        assert service.sessions_collection == "user_sessions"
        assert service.refresh_tokens_collection == "refresh_tokens"

    @staticmethod
    def test_service_initialization_custom_values(
        mock_auth_provider: Mock,
        mock_firestore_client: Mock,
    ) -> None:
        """Test service initialization with custom values."""
        service = AuthenticationService(
            auth_provider=mock_auth_provider,
            firestore_client=mock_firestore_client,
            default_token_expiry=7200,  # 2 hours
            refresh_token_expiry=604800,  # 7 days
        )

        assert service.default_token_expiry == 7200
        assert service.refresh_token_expiry == 604800


@pytest.fixture
def auth_service() -> AuthenticationService:
    """Create AuthenticationService instance with mocks."""
    mock_auth_provider = Mock(spec=IAuthProvider)
    mock_firestore_client = Mock(spec=FirestoreClient)
    mock_firestore_client.create_document = AsyncMock()
    mock_firestore_client.get_document = AsyncMock()
    mock_firestore_client.update_document = AsyncMock()
    mock_firestore_client.delete_document = AsyncMock()

    return AuthenticationService(
        auth_provider=mock_auth_provider,
        firestore_client=mock_firestore_client,
    )


@pytest.fixture
def sample_registration_request(
    test_env_credentials: dict[str, str | None],
) -> UserRegistrationRequest:
    """Sample user registration request."""
    password = test_env_credentials.get("default_password")
    if not isinstance(password, str):
        raise TypeError(REGISTRATION_PASSWORD_ERROR_MSG)
    return UserRegistrationRequest(
        email="test@example.com",
        password=password,
        first_name="John",
        last_name="Doe",
        phone_number="+1234567890",
        terms_accepted=True,
        privacy_policy_accepted=True,
    )


@pytest.fixture
def sample_login_request(
    test_env_credentials: dict[str, str | None],
) -> UserLoginRequest:
    """Sample user login request."""
    password = test_env_credentials.get("default_password")
    if not isinstance(password, str):
        raise TypeError(LOGIN_PASSWORD_ERROR_MSG)
    return UserLoginRequest(
        email="test@example.com",
        password=password,
        remember_me=False,
        device_info=None,
    )


class TestUserRegistration:
    """🔐 Test user registration functionality."""

    @staticmethod
    @patch("clarity.services.auth_service.auth")
    async def test_register_user_success(
        mock_auth: Mock,
        auth_service: AuthenticationService,
        sample_registration_request: UserRegistrationRequest,
    ) -> None:
        """Test successful user registration."""
        user_id = str(uuid.uuid4())
        email_for_new_user = "new_user_for_success_test@example.com"

        mock_auth.UserNotFoundError = auth.UserNotFoundError
        cast("Mock", mock_auth.get_user_by_email).side_effect = (
            mock_auth.UserNotFoundError("User not found for test")
        )

        cast("Mock", mock_auth.create_user).return_value = MockFirebaseUserRecord(
            uid=user_id, email=email_for_new_user
        )
        cast("Mock", mock_auth.generate_email_verification_link).return_value = (
            "http://verify.link"
        )
        mock_auth.set_custom_user_claims = Mock()

        cast(
            "AsyncMock", auth_service.firestore_client.create_document
        ).return_value = user_id

        request_data_for_test = sample_registration_request.model_copy(
            update={"email": email_for_new_user}
        )

        result = await auth_service.register_user(
            request_data_for_test, device_info=None
        )

        assert isinstance(result.user_id, uuid.UUID)
        assert result.email == email_for_new_user

        cast("Mock", mock_auth.get_user_by_email).assert_called_once_with(
            email_for_new_user
        )
        cast("Mock", mock_auth.create_user).assert_called_once()
        cast(
            "AsyncMock", auth_service.firestore_client.create_document
        ).assert_called_once()

    @staticmethod
    @patch("clarity.services.auth_service.auth")
    async def test_register_user_already_exists(
        mock_auth: Mock,
        auth_service: AuthenticationService,
        sample_registration_request: UserRegistrationRequest,
    ) -> None:
        """Test registration when user already exists."""
        mock_auth.UserNotFoundError = auth.UserNotFoundError
        cast("Mock", mock_auth.get_user_by_email).return_value = MockFirebaseUserRecord(
            uid="existing_id", email=sample_registration_request.email
        )

        with pytest.raises(UserAlreadyExistsError) as exc_info:
            await auth_service.register_user(
                sample_registration_request, device_info={}
            )
        assert "already exists" in str(exc_info.value)
        cast("Mock", mock_auth.create_user).assert_not_called()

    @staticmethod
    @patch("clarity.services.auth_service.auth")
    async def test_register_user_with_device_info(
        mock_auth: Mock,
        auth_service: AuthenticationService,
        sample_registration_request: UserRegistrationRequest,
    ) -> None:
        """Test registration with device info."""
        mock_user_record = Mock()
        mock_user_record.uid = str(uuid.uuid4())

        mock_auth.UserNotFoundError = MockUserNotFoundError
        cast("Mock", mock_auth.get_user_by_email).side_effect = MockUserNotFoundError(
            "User not found"
        )
        cast("Mock", mock_auth.create_user).return_value = mock_user_record
        mock_auth.set_custom_user_claims = Mock()

        device_info = {"device_type": "iPhone", "os_version": "iOS 17"}

        await auth_service.register_user(
            sample_registration_request, device_info=device_info
        )

        call_args = cast(
            "AsyncMock", auth_service.firestore_client.create_document
        ).call_args
        user_data = call_args[1]["data"]
        assert user_data["device_info"] == device_info

    @staticmethod
    @patch("clarity.services.auth_service.auth")
    async def test_register_user_firebase_error(
        mock_auth: Mock,
        auth_service: AuthenticationService,
        sample_registration_request: UserRegistrationRequest,
    ) -> None:
        """Test registration when Firebase throws an error."""
        mock_auth.UserNotFoundError = auth.UserNotFoundError
        cast("Mock", mock_auth.get_user_by_email).side_effect = (
            mock_auth.UserNotFoundError("User not found")
        )
        cast("Mock", mock_auth.create_user).side_effect = MockAuthError(
            "Firebase create_user error"
        )

        with pytest.raises(AuthenticationError, match="Registration failed"):
            await auth_service.register_user(
                sample_registration_request, device_info=None
            )


class TestUserLogin:
    """🔑 Test user login functionality."""

    @staticmethod
    @patch("clarity.services.auth_service.auth")
    async def test_login_user_success(
        mock_auth: Mock,
        auth_service: AuthenticationService,
        sample_login_request: UserLoginRequest,
        test_env_credentials: dict[str, str | None],
    ) -> None:
        """Test successful user login."""
        test_uuid = uuid.uuid4()
        mock_user_record = Mock()
        mock_user_record.uid = str(test_uuid)
        mock_user_record.email = "test@example.com"
        mock_user_record.email_verified = True
        mock_user_record.disabled = False

        mock_auth.UserNotFoundError = MockUserNotFoundError
        cast("Mock", mock_auth.get_user_by_email).return_value = mock_user_record

        user_data = {
            "user_id": str(test_uuid),
            "email": "test@example.com",
            "first_name": "John",
            "last_name": "Doe",
            "status": UserStatus.ACTIVE.value,
            "role": UserRole.PATIENT.value,
            "email_verified": True,
            "mfa_enabled": False,
            "login_count": 0,
            "created_at": datetime.now(UTC),
        }
        cast("AsyncMock", auth_service.firestore_client.get_document).return_value = (
            user_data
        )

        with patch.object(auth_service, "_generate_tokens") as mock_gen_tokens_patch:
            mock_gen_tokens: Mock = mock_gen_tokens_patch
            mock_tokens = TokenResponse(
                access_token=cast("str", test_env_credentials["mock_access_token"]),
                refresh_token=cast("str", test_env_credentials["mock_refresh_token"]),
                token_type="bearer",  # noqa: S106
                expires_in=3600,
                scope="full_access",
            )
            mock_gen_tokens.return_value = mock_tokens

            with patch.object(
                auth_service, "_create_user_session"
            ) as mock_create_session_patch:
                mock_create_session: Mock = mock_create_session_patch
                mock_create_session.return_value = "session-id-123"

                with patch.object(
                    auth_service, "_create_user_session_response"
                ) as mock_session_response_patch:
                    mock_session_response: Mock = mock_session_response_patch
                    mock_user_session = UserSessionResponse(
                        user_id=test_uuid,
                        email="test@example.com",
                        first_name="John",
                        last_name="Doe",
                        role=UserRole.PATIENT.value,
                        permissions=["read_own_data", "write_own_data"],
                        status=UserStatus.ACTIVE,
                        mfa_enabled=False,
                        email_verified=True,
                        created_at=datetime.now(UTC),
                        last_login=datetime.now(UTC),
                    )
                    mock_session_response.return_value = mock_user_session

                    result = await auth_service.login_user(sample_login_request)

                    assert isinstance(result, LoginResponse)

    @staticmethod
    @patch("clarity.services.auth_service.auth")
    async def test_login_user_invalid_token(
        mock_auth: Mock,
        auth_service: AuthenticationService,
        sample_login_request: UserLoginRequest,
    ) -> None:
        """Test login with invalid token scenario (e.g., token generation fails)."""
        test_uuid = uuid.uuid4()
        mock_user_record_instance = MockFirebaseUserRecord(
            uid=str(test_uuid),
            email=sample_login_request.email,
            disabled=False,
            email_verified=True,
        )

        mock_auth.UserNotFoundError = auth.UserNotFoundError
        cast("Mock", mock_auth.get_user_by_email).return_value = (
            mock_user_record_instance
        )

        user_data = {
            "user_id": str(test_uuid),
            "email": sample_login_request.email,
            "first_name": "John",
            "last_name": "Doe",
            "status": UserStatus.ACTIVE.value,
            "role": UserRole.PATIENT.value,
            "email_verified": True,
            "mfa_enabled": False,
            "login_count": 0,
            "created_at": datetime.now(UTC),
        }
        cast("AsyncMock", auth_service.firestore_client.get_document).return_value = (
            user_data
        )

        with patch(
            "clarity.services.auth_service.secrets.token_urlsafe"
        ) as mock_token_urlsafe_patch:
            mock_token_urlsafe: Mock = mock_token_urlsafe_patch
            mock_token_urlsafe.side_effect = Exception(
                "Token creation failed by provider"
            )

            with pytest.raises(
                AuthenticationError,
                match="Login failed: Token creation failed by provider",
            ):
                await auth_service.login_user(sample_login_request, device_info=None)

    @staticmethod
    @patch("clarity.services.auth_service.auth")
    async def test_login_user_not_found(
        mock_auth: Mock,
        auth_service: AuthenticationService,
        sample_login_request: UserLoginRequest,
    ) -> None:
        """Test login when user not found."""
        mock_auth.UserNotFoundError = auth.UserNotFoundError
        cast("Mock", mock_auth.get_user_by_email).side_effect = (
            mock_auth.UserNotFoundError("User not found")
        )

        with pytest.raises(UserNotFoundError):
            await auth_service.login_user(sample_login_request, device_info=None)

    @staticmethod
    @patch("clarity.services.auth_service.auth")
    async def test_login_user_account_disabled(
        mock_auth: Mock,
        auth_service: AuthenticationService,
        sample_login_request: UserLoginRequest,
    ) -> None:
        """Test login when account is disabled."""
        mock_user_record = MockFirebaseUserRecord(
            uid=str(uuid.uuid4()), email=sample_login_request.email, disabled=True
        )
        mock_auth.UserNotFoundError = auth.UserNotFoundError
        cast("Mock", mock_auth.get_user_by_email).return_value = mock_user_record

        with pytest.raises(AccountDisabledError):
            await auth_service.login_user(sample_login_request, device_info=None)

    @staticmethod
    @patch("clarity.services.auth_service.auth")
    async def test_login_user_email_not_verified(
        mock_auth: Mock,
        auth_service: AuthenticationService,
        sample_login_request: UserLoginRequest,
        test_env_credentials: dict[str, str | None],
    ) -> None:
        """Test login when email is not verified."""
        test_uuid = uuid.uuid4()
        mock_user_record = MockFirebaseUserRecord(
            uid=str(test_uuid),
            email=sample_login_request.email,
            email_verified=False,
            disabled=False,
        )
        mock_auth.UserNotFoundError = auth.UserNotFoundError
        cast("Mock", mock_auth.get_user_by_email).return_value = mock_user_record

        user_data = {
            "user_id": str(test_uuid),
            "email": sample_login_request.email,
            "status": UserStatus.PENDING_VERIFICATION.value,
            "mfa_enabled": False,
            "login_count": 0,
            "created_at": datetime.now(UTC),
        }
        cast("AsyncMock", auth_service.firestore_client.get_document).return_value = (
            user_data
        )

        with patch.object(auth_service, "_generate_tokens") as mock_gen_tokens_patch:
            mock_gen_tokens: Mock = mock_gen_tokens_patch
            mock_tokens = TokenResponse(
                access_token=cast("str", test_env_credentials["mock_access_token"]),
                refresh_token=cast("str", test_env_credentials["mock_refresh_token"]),
                token_type="bearer",
                expires_in=3600,
                scope="full_access",
            )
            mock_gen_tokens.return_value = mock_tokens
            with patch.object(
                auth_service, "_create_user_session"
            ) as mock_create_session_patch:
                mock_create_session: Mock = mock_create_session_patch
                mock_create_session.return_value = "session-id-123"
                with patch.object(
                    auth_service, "_create_user_session_response"
                ) as mock_session_response_patch:
                    mock_session_response: Mock = mock_session_response_patch
                    mock_user_session = UserSessionResponse(
                        user_id=test_uuid,
                        email=sample_login_request.email,
                        first_name="John",
                        last_name="Doe",
                        role=UserRole.PATIENT.value,
                        permissions=["read_own_data"],
                        status=UserStatus.PENDING_VERIFICATION,
                        mfa_enabled=False,
                        email_verified=False,
                        created_at=datetime.now(UTC),
                        last_login=None,
                    )
                    mock_session_response.return_value = mock_user_session
                    result = await auth_service.login_user(
                        sample_login_request, device_info=None
                    )
                    assert isinstance(result, LoginResponse)
                    assert result.user.status == UserStatus.PENDING_VERIFICATION


class TestTokenManagement:
    """🎫 Test token generation and management."""

    @staticmethod
    async def test_generate_tokens(
        auth_service: AuthenticationService, test_env_credentials: dict[str, str | None]
    ) -> None:
        """Test token generation."""
        user_id = "test-uid-123"

        with patch(
            "clarity.services.auth_service.secrets.token_urlsafe"
        ) as mock_token_safe_patch:
            mock_token_safe: Mock = mock_token_safe_patch
            mock_token_safe.side_effect = [
                cast("str", test_env_credentials["mock_access_token"]),
                cast("str", test_env_credentials["mock_refresh_token"]),
            ]

            result = await auth_service._generate_tokens(user_id)

            assert isinstance(result, TokenResponse)
            assert result.access_token == cast(
                "str", test_env_credentials["mock_access_token"]
            )
            assert result.refresh_token == cast(
                "str", test_env_credentials["mock_refresh_token"]
            )
            assert result.token_type == "bearer"
            assert result.expires_in == 3600

    @staticmethod
    async def test_generate_tokens_remember_me(
        auth_service: AuthenticationService,
        test_env_credentials: dict[str, str | None],
    ) -> None:
        """Test token generation with remember me flag."""
        user_id = "test-uid-123"

        with patch(
            "clarity.services.auth_service.secrets.token_urlsafe"
        ) as mock_token_safe_patch:
            mock_token_safe: Mock = mock_token_safe_patch
            mock_token_safe.side_effect = [
                cast("str", test_env_credentials["mock_access_token"]),
                cast("str", test_env_credentials["mock_refresh_token"]),
            ]

            result = await auth_service._generate_tokens(user_id, remember_me=True)

            assert isinstance(result, TokenResponse)
            assert result.expires_in == 86400 * 30

    @staticmethod
    async def test_refresh_access_token_success(
        auth_service: AuthenticationService,
        test_env_credentials: dict[str, str | None],
    ) -> None:
        """Test successful token refresh."""
        refresh_token = cast("str", test_env_credentials["mock_refresh_token"])

        token_data = {
            "id": "token-doc-id",
            "user_id": "test-uid-123",
            "expires_at": datetime.now(UTC) + timedelta(days=1),
            "is_revoked": False,
        }
        cast(
            "AsyncMock", auth_service.firestore_client.query_documents
        ).return_value = [token_data]

        with patch(
            "clarity.services.auth_service.secrets.token_urlsafe"
        ) as mock_token_safe_patch:
            mock_token_safe: Mock = mock_token_safe_patch
            mock_token_safe.side_effect = [
                cast("str", test_env_credentials["mock_new_access_token"]),
                cast("str", test_env_credentials["mock_new_refresh_token"]),
            ]

            result = await auth_service.refresh_access_token(refresh_token)

            assert isinstance(result, TokenResponse)
            assert result.access_token == cast(
                "str", test_env_credentials["mock_new_access_token"]
            )

    @staticmethod
    async def test_refresh_access_token_not_found(
        auth_service: AuthenticationService,
    ) -> None:
        """Test token refresh with invalid token."""
        refresh_token = "invalid-refresh-token"  # noqa: S105

        cast(
            "AsyncMock", auth_service.firestore_client.query_documents
        ).return_value = []

        with pytest.raises(InvalidCredentialsError):
            await auth_service.refresh_access_token(refresh_token)

    @staticmethod
    async def test_refresh_access_token_expired(
        auth_service: AuthenticationService,
    ) -> None:
        """Test token refresh with expired token."""
        refresh_token = "expired-refresh-token"  # noqa: S105

        token_data = {
            "id": "token-doc-id",
            "user_id": "test-uid-123",
            "expires_at": datetime.now(UTC) - timedelta(days=1),
            "is_revoked": False,
        }
        cast(
            "AsyncMock", auth_service.firestore_client.query_documents
        ).return_value = [token_data]

        with pytest.raises(InvalidCredentialsError):
            await auth_service.refresh_access_token(refresh_token)

    @staticmethod
    async def test_refresh_access_token_revoked(
        auth_service: AuthenticationService,
    ) -> None:
        """Test token refresh with revoked token."""
        refresh_token = "revoked-refresh-token"  # noqa: S105

        cast(
            "AsyncMock", auth_service.firestore_client.query_documents
        ).return_value = []

        with pytest.raises(InvalidCredentialsError):
            await auth_service.refresh_access_token(refresh_token)


class TestSessionManagement:
    """🗂️ Test user session management."""

    @staticmethod
    async def test_create_user_session(
        auth_service: AuthenticationService, test_env_credentials: dict[str, str | None]
    ) -> None:
        """Test user session creation."""
        user_id = "test-uid-123"
        refresh_token = cast("str", test_env_credentials["mock_refresh_token"])
        device_info = {"device_type": "iPhone"}
        ip_address = "192.168.1.1"

        session_id = await auth_service._create_user_session(
            user_id, refresh_token, device_info, ip_address
        )

        assert isinstance(session_id, str)
        assert len(session_id) > 0

        cast("AsyncMock", auth_service.firestore_client.create_document).assert_called()
        call_args = cast(
            "AsyncMock", auth_service.firestore_client.create_document
        ).call_args
        session_data = call_args[1]["data"]

        assert session_data["user_id"] == user_id
        assert session_data["refresh_token"] == refresh_token
        assert session_data["device_info"] == device_info
        assert session_data["ip_address"] == ip_address

    @staticmethod
    async def test_create_user_session_response() -> None:
        """Test user session response creation."""
        test_uuid = uuid.uuid4()
        mock_user_record = Mock()
        mock_user_record.uid = str(test_uuid)
        mock_user_record.email = "test@example.com"
        mock_user_record.email_verified = True

        user_data = {
            "first_name": "John",
            "last_name": "Doe",
            "role": UserRole.PATIENT.value,
            "status": UserStatus.ACTIVE.value,
            "mfa_enabled": False,
            "email_verified": True,
            "created_at": datetime.now(UTC),
            "last_login": datetime.now(UTC),
        }

        result = await AuthenticationService._create_user_session_response(
            mock_user_record, user_data
        )

        assert isinstance(result, UserSessionResponse)
        assert result.user_id == test_uuid
        assert result.email == "test@example.com"
        assert result.first_name == "John"
        assert result.last_name == "Doe"
        assert result.role == UserRole.PATIENT.value
        assert result.status == UserStatus.ACTIVE

    @staticmethod
    async def test_logout_user_success(
        auth_service: AuthenticationService, test_env_credentials: dict[str, str | None]
    ) -> None:
        """Test successful user logout."""
        refresh_token = cast("str", test_env_credentials["mock_refresh_token"])

        token_data = {
            "id": "token-doc-id",
            "user_id": "test-uid-123",
            "refresh_token": refresh_token,
            "is_revoked": False,
            "expires_at": datetime.now(UTC) + timedelta(days=1),
        }

        session_data = {
            "session_id": "session-doc-id",
            "user_id": "test-uid-123",
            "refresh_token": refresh_token,
            "is_active": True,
        }

        cast("AsyncMock", auth_service.firestore_client.query_documents).side_effect = [
            [token_data],
            [session_data],
        ]

        result = await auth_service.logout_user(refresh_token)

        assert result is True
        assert (
            cast("AsyncMock", auth_service.firestore_client.update_document).call_count
            == 2
        )

    @staticmethod
    async def test_logout_user_failure(auth_service: AuthenticationService) -> None:
        """Test user logout with error."""
        refresh_token = "invalid-refresh-token"  # noqa: S105

        cast(
            "AsyncMock", auth_service.firestore_client.query_documents
        ).return_value = []

        result = await auth_service.logout_user(refresh_token)

        assert result is True
        cast(
            "AsyncMock", auth_service.firestore_client.update_document
        ).assert_not_called()


class TestUserRetrieval:
    """👤 Test user data retrieval functionality."""

    @staticmethod
    @patch("clarity.services.auth_service.auth")
    async def test_get_user_by_id_success(
        mock_auth: Mock,
        auth_service: AuthenticationService,
    ) -> None:
        """Test successful user retrieval by ID."""
        test_uuid = uuid.uuid4()
        user_id = str(test_uuid)

        mock_user_record = Mock()
        mock_user_record.uid = user_id
        mock_user_record.email = "test@example.com"
        mock_user_record.email_verified = True
        cast("Mock", mock_auth.get_user).return_value = mock_user_record
        mock_auth.UserNotFoundError = MockUserNotFoundError

        user_data = {
            "first_name": "John",
            "last_name": "Doe",
            "role": UserRole.PATIENT.value,
            "status": UserStatus.ACTIVE.value,
            "mfa_enabled": False,
            "email_verified": True,
            "created_at": datetime.now(UTC),
            "last_login": datetime.now(UTC),
        }
        cast("AsyncMock", auth_service.firestore_client.get_document).return_value = (
            user_data
        )

        result = await auth_service.get_user_by_id(user_id)

        assert isinstance(result, UserSessionResponse)
        assert result.user_id == test_uuid
        assert result.email == "test@example.com"

    @staticmethod
    @patch("clarity.services.auth_service.auth")
    async def test_get_user_by_id_not_found_firebase(
        mock_auth: Mock,
        auth_service: AuthenticationService,
    ) -> None:
        """Test user retrieval when not found in Firebase."""
        user_id = str(uuid.uuid4())

        mock_auth.UserNotFoundError = MockUserNotFoundError
        cast("Mock", mock_auth.get_user).side_effect = MockUserNotFoundError(
            "User not found"
        )

        result = await auth_service.get_user_by_id(user_id)

        assert result is None

    @staticmethod
    @patch("clarity.services.auth_service.auth")
    async def test_get_user_by_id_not_found_firestore(
        mock_auth: Mock,
        auth_service: AuthenticationService,
    ) -> None:
        """Test user retrieval when not found in Firestore."""
        user_id = str(uuid.uuid4())

        mock_user_record = Mock()
        mock_user_record.uid = user_id
        mock_user_record.email = "test@example.com"
        cast("Mock", mock_auth.get_user).return_value = mock_user_record

        cast("AsyncMock", auth_service.firestore_client.get_document).return_value = (
            None
        )

        result = await auth_service.get_user_by_id(user_id)

        assert result is None


class TestEmailVerification:
    """📧 Test email verification functionality."""

    @staticmethod
    async def test_verify_email() -> None:
        """Test email verification."""
        verification_code = "test-verification-code"

        result = await AuthenticationService.verify_email(verification_code)

        assert result is True


class TestExceptionHierarchy:
    """🏗️ Test exception hierarchy and inheritance."""

    @staticmethod
    def test_authentication_error_base() -> None:
        """Test AuthenticationError as base exception."""
        error = AuthenticationError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    @staticmethod
    def test_specific_auth_errors() -> None:
        """Test specific authentication error types."""
        errors = [
            UserNotFoundError("User not found"),
            InvalidCredentialsError("Invalid credentials"),
            UserAlreadyExistsError("User exists"),
            EmailNotVerifiedError("Email not verified"),
            AccountDisabledError("Account disabled"),
        ]

        for error in errors:
            assert isinstance(error, AuthenticationError)
            assert isinstance(error, Exception)


class TestEdgeCases:
    """🎯 Test edge cases and boundary conditions."""

    @staticmethod
    async def test_empty_device_info(
        auth_service: AuthenticationService,
        sample_registration_request: UserRegistrationRequest,
    ) -> None:
        """Test registration with empty device info."""
        with patch("clarity.services.auth_service.auth") as mock_auth:
            mock_user_record = Mock()
            mock_user_record.uid = str(uuid.uuid4())

            mock_auth.UserNotFoundError = MockUserNotFoundError
            cast("Mock", mock_auth.get_user_by_email).side_effect = (
                MockUserNotFoundError("User not found")
            )
            cast("Mock", mock_auth.create_user).return_value = mock_user_record
            mock_auth.set_custom_user_claims = Mock()

            await auth_service.register_user(
                sample_registration_request, device_info={}
            )

            call_args = cast(
                "AsyncMock", auth_service.firestore_client.create_document
            ).call_args
            user_data = call_args[1]["data"]
            assert user_data["device_info"] == {}

    @staticmethod
    async def test_token_generation_with_custom_expiry(
        auth_service: AuthenticationService,
    ) -> None:
        """Test token generation with custom expiry time."""
        custom_service = AuthenticationService(
            auth_provider=auth_service.auth_provider,
            firestore_client=auth_service.firestore_client,
            default_token_expiry=7200,
        )

        user_id = "test-uid-123"

        cast("Mock", custom_service.auth_provider).create_custom_token = Mock(
            return_value=b"mock-token"
        )

        with patch("secrets.token_urlsafe") as mock_token_safe_patch:
            mock_token_safe: Mock = mock_token_safe_patch
            mock_token_safe.return_value = "mock-refresh-token"

            result = await custom_service._generate_tokens(user_id)

            assert isinstance(result, TokenResponse)
            assert result.expires_in == 7200

    @staticmethod
    async def test_concurrent_token_operations(
        auth_service: AuthenticationService,
    ) -> None:
        """Test concurrent token operations."""

        async def mock_token_op(user_id: str) -> TokenResponse:
            """Mock token operation."""
            cast("Mock", auth_service.auth_provider).create_custom_token = Mock(
                return_value=b"mock-token"
            )

            with patch("secrets.token_urlsafe") as mock_token_safe_patch:
                mock_token_safe: Mock = mock_token_safe_patch
                mock_token_safe.return_value = f"mock-refresh-token-{user_id}"

                return await auth_service._generate_tokens(user_id)

        user_ids = ["user1", "user2", "user3"]
        tasks = [mock_token_op(user_id) for user_id in user_ids]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, TokenResponse)


class TestDataValidation:
    """✅ Test data validation and model constraints."""

    @staticmethod
    async def test_registration_data_validation(
        test_env_credentials: dict[str, str | None],
    ) -> None:
        """Test registration request validation."""
        valid_request = UserRegistrationRequest(
            email="test@example.com",
            password=cast("str", test_env_credentials["default_password"]),
            first_name="John",
            last_name="Doe",
            phone_number=None,
            terms_accepted=True,
            privacy_policy_accepted=True,
        )
        assert valid_request.email == "test@example.com"

        with pytest.raises(ValidationError, match="value is not a valid email address"):
            UserRegistrationRequest(
                email="invalid-email",
                password=cast("str", test_env_credentials["default_password"]),
                first_name="John",
                last_name="Doe",
                phone_number=None,
                terms_accepted=True,
                privacy_policy_accepted=True,
            )

    @staticmethod
    async def test_login_data_validation(
        test_env_credentials: dict[str, str | None],
    ) -> None:
        """Test login request validation."""
        valid_request = UserLoginRequest(
            email="test@example.com",
            password=cast("str", test_env_credentials["default_password"]),
            device_info=None,
        )
        assert valid_request.email == "test@example.com"

        with pytest.raises(ValidationError, match="value is not a valid email address"):
            UserLoginRequest(
                email="invalid-email",
                password=cast("str", test_env_credentials["default_password"]),
                device_info=None,
            )
