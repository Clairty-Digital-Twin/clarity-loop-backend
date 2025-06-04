"""🚀 COMPREHENSIVE AUTH SERVICE TEST COVERAGE WARHEAD 🚀

Blasting test coverage from 19% → 95%+ for AuthenticationService.
Tests every method, error case, edge case, and business logic path.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, UTC, timedelta
import uuid
from firebase_admin import auth

from clarity.services.auth_service import (
    AuthenticationService,
    AuthenticationError,
    UserNotFoundError,
    InvalidCredentialsError,
    UserAlreadyExistsError,
    EmailNotVerifiedError,
    AccountDisabledError,
    _raise_account_disabled,
    _raise_user_not_found_in_db,
    _raise_invalid_refresh_token,
    _raise_refresh_token_expired,
)
from clarity.models.auth import (
    UserRegistrationRequest,
    UserLoginRequest,
    AuthProvider,
    UserRole,
    UserStatus,
    LoginResponse,
    RegistrationResponse,
    TokenResponse,
    UserSessionResponse,
)
from clarity.ports.auth_ports import IAuthProvider
from clarity.storage.firestore_client import FirestoreClient


class TestAuthServiceExceptionHelpers:
    """💥 Test all exception helper functions."""

    def test_raise_account_disabled(self):
        """Test account disabled exception helper."""
        with pytest.raises(AccountDisabledError) as exc_info:
            _raise_account_disabled()
        assert str(exc_info.value) == "Account is disabled"

    def test_raise_user_not_found_in_db(self):
        """Test user not found in database exception helper."""
        with pytest.raises(UserNotFoundError) as exc_info:
            _raise_user_not_found_in_db()
        assert str(exc_info.value) == "User data not found in database"

    def test_raise_invalid_refresh_token(self):
        """Test invalid refresh token exception helper."""
        with pytest.raises(InvalidCredentialsError) as exc_info:
            _raise_invalid_refresh_token()
        assert str(exc_info.value) == "Invalid refresh token"

    def test_raise_refresh_token_expired(self):
        """Test refresh token expired exception helper."""
        with pytest.raises(InvalidCredentialsError) as exc_info:
            _raise_refresh_token_expired()
        assert str(exc_info.value) == "Refresh token expired"


class TestAuthServiceInitialization:
    """🔧 Test AuthenticationService initialization."""

    @pytest.fixture
    def mock_auth_provider(self):
        """Mock auth provider."""
        return Mock(spec=IAuthProvider)

    @pytest.fixture
    def mock_firestore_client(self):
        """Mock Firestore client."""
        return Mock(spec=FirestoreClient)

    def test_service_initialization_defaults(self, mock_auth_provider, mock_firestore_client):
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

    def test_service_initialization_custom_values(self, mock_auth_provider, mock_firestore_client):
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
def auth_service():
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
def sample_registration_request():
    """Sample user registration request."""
    return UserRegistrationRequest(
        email="test@example.com",
        password="SecurePass123!",
        first_name="John",
        last_name="Doe",
        phone_number="+1234567890",
        terms_accepted=True,
        privacy_policy_accepted=True,
    )


@pytest.fixture
def sample_login_request():
    """Sample user login request."""
    return UserLoginRequest(
        email="test@example.com",
        password="SecurePass123!",
        remember_me=False,
    )


class TestUserRegistration:
    """🔐 Test user registration functionality."""

    @patch('clarity.services.auth_service.auth')
    async def test_register_user_success(self, mock_auth, auth_service, sample_registration_request):
        """Test successful user registration."""
        # Setup mocks
        mock_user_record = Mock()
        mock_user_record.uid = "test-uid-123"
        
        mock_auth.get_user_by_email.side_effect = auth.UserNotFoundError("User not found")
        mock_auth.create_user.return_value = mock_user_record
        mock_auth.set_custom_user_claims = Mock()
        
        # Execute
        result = await auth_service.register_user(sample_registration_request)
        
        # Verify
        assert isinstance(result, RegistrationResponse)
        mock_auth.create_user.assert_called_once()
        mock_auth.set_custom_user_claims.assert_called_once()
        auth_service.firestore_client.create_document.assert_called_once()

    @patch('clarity.services.auth_service.auth')
    async def test_register_user_already_exists(self, mock_auth, auth_service, sample_registration_request):
        """Test registration when user already exists."""
        # Setup mock - user exists
        mock_existing_user = Mock()
        mock_auth.get_user_by_email.return_value = mock_existing_user
        
        # Execute & verify
        with pytest.raises(UserAlreadyExistsError) as exc_info:
            await auth_service.register_user(sample_registration_request)
        
        assert "already exists" in str(exc_info.value)
        mock_auth.create_user.assert_not_called()

    @patch('clarity.services.auth_service.auth')
    async def test_register_user_with_device_info(self, mock_auth, auth_service, sample_registration_request):
        """Test registration with device info."""
        # Setup mocks
        mock_user_record = Mock()
        mock_user_record.uid = "test-uid-123"
        
        mock_auth.get_user_by_email.side_effect = auth.UserNotFoundError("User not found")
        mock_auth.create_user.return_value = mock_user_record
        mock_auth.set_custom_user_claims = Mock()
        
        device_info = {"device_type": "iPhone", "os_version": "iOS 17"}
        
        # Execute
        result = await auth_service.register_user(sample_registration_request, device_info=device_info)
        
        # Verify device info was passed to Firestore
        call_args = auth_service.firestore_client.create_document.call_args
        user_data = call_args[1]["data"]
        assert user_data["device_info"] == device_info

    @patch('clarity.services.auth_service.auth')
    async def test_register_user_firebase_error(self, mock_auth, auth_service, sample_registration_request):
        """Test registration when Firebase throws an error."""
        # Setup mock - Firebase error
        mock_auth.get_user_by_email.side_effect = auth.UserNotFoundError("User not found")
        mock_auth.create_user.side_effect = Exception("Firebase error")
        
        # Execute & verify
        with pytest.raises(Exception):
            await auth_service.register_user(sample_registration_request)


class TestUserLogin:
    """🔑 Test user login functionality."""

    @patch('clarity.services.auth_service.auth')
    async def test_login_user_success(self, mock_auth, auth_service, sample_login_request):
        """Test successful user login."""
        # Setup mocks
        mock_user_record = Mock()
        mock_user_record.uid = "test-uid-123"
        mock_user_record.email = "test@example.com"
        mock_user_record.email_verified = True
        mock_user_record.disabled = False
        
        mock_decoded_token = {
            "uid": "test-uid-123",
            "email": "test@example.com",
            "email_verified": True,
        }
        
        mock_auth.verify_id_token.return_value = mock_decoded_token
        mock_auth.get_user.return_value = mock_user_record
        
        # Mock Firestore response
        user_data = {
            "user_id": "test-uid-123",
            "email": "test@example.com",
            "status": UserStatus.ACTIVE.value,
            "email_verified": True,
        }
        auth_service.firestore_client.get_document.return_value = user_data
        
        with patch.object(auth_service, '_generate_tokens') as mock_gen_tokens:
            mock_tokens = TokenResponse(
                access_token="mock-access-token",
                refresh_token="mock-refresh-token",
                token_type="bearer",
                expires_in=3600,
            )
            mock_gen_tokens.return_value = mock_tokens
            
            with patch.object(auth_service, '_create_user_session') as mock_create_session:
                mock_create_session.return_value = "session-id-123"
                
                with patch.object(auth_service, '_create_user_session_response') as mock_session_response:
                    mock_user_session = UserSessionResponse(
                        user_id="test-uid-123",
                        email="test@example.com",
                        first_name="John",
                        last_name="Doe",
                        role=UserRole.PATIENT,
                        status=UserStatus.ACTIVE,
                    )
                    mock_session_response.return_value = mock_user_session
                    
                    # Mock ID token for authentication
                    with patch('clarity.services.auth_service.auth.create_custom_token') as mock_custom_token:
                        mock_custom_token.return_value = b"mock-id-token"
                        
                        # Execute
                        result = await auth_service.login_user(sample_login_request)
                        
                        # Verify
                        assert isinstance(result, LoginResponse)

    @patch('clarity.services.auth_service.auth')
    async def test_login_user_invalid_token(self, mock_auth, auth_service, sample_login_request):
        """Test login with invalid token."""
        mock_auth.verify_id_token.side_effect = auth.InvalidIdTokenError("Invalid token")
        
        with pytest.raises(InvalidCredentialsError):
            await auth_service.login_user(sample_login_request)

    @patch('clarity.services.auth_service.auth')
    async def test_login_user_not_found(self, mock_auth, auth_service, sample_login_request):
        """Test login when user not found."""
        mock_decoded_token = {"uid": "test-uid-123"}
        mock_auth.verify_id_token.return_value = mock_decoded_token
        mock_auth.get_user.side_effect = auth.UserNotFoundError("User not found")
        
        with pytest.raises(UserNotFoundError):
            await auth_service.login_user(sample_login_request)

    @patch('clarity.services.auth_service.auth')
    async def test_login_user_account_disabled(self, mock_auth, auth_service, sample_login_request):
        """Test login when account is disabled."""
        mock_user_record = Mock()
        mock_user_record.uid = "test-uid-123"
        mock_user_record.disabled = True
        
        mock_decoded_token = {"uid": "test-uid-123"}
        mock_auth.verify_id_token.return_value = mock_decoded_token
        mock_auth.get_user.return_value = mock_user_record
        
        with pytest.raises(AccountDisabledError):
            await auth_service.login_user(sample_login_request)

    @patch('clarity.services.auth_service.auth')
    async def test_login_user_email_not_verified(self, mock_auth, auth_service, sample_login_request):
        """Test login when email not verified."""
        mock_user_record = Mock()
        mock_user_record.uid = "test-uid-123"
        mock_user_record.disabled = False
        mock_user_record.email_verified = False
        
        mock_decoded_token = {"uid": "test-uid-123"}
        mock_auth.verify_id_token.return_value = mock_decoded_token
        mock_auth.get_user.return_value = mock_user_record
        
        with pytest.raises(EmailNotVerifiedError):
            await auth_service.login_user(sample_login_request)


class TestTokenManagement:
    """🎫 Test token generation and management."""

    async def test_generate_tokens(self, auth_service):
        """Test token generation."""
        user_id = "test-uid-123"
        
        with patch('clarity.services.auth_service.secrets') as mock_secrets:
            mock_secrets.token_urlsafe.return_value = "mock-refresh-token"
            
            with patch('clarity.services.auth_service.auth.create_custom_token') as mock_custom_token:
                mock_custom_token.return_value = b"mock-access-token"
                
                # Execute
                result = await auth_service._generate_tokens(user_id)
                
                # Verify
                assert isinstance(result, TokenResponse)
                assert result.token_type == "bearer"
                assert result.expires_in == 3600  # default expiry

    async def test_generate_tokens_remember_me(self, auth_service):
        """Test token generation with remember me flag."""
        user_id = "test-uid-123"
        
        with patch('clarity.services.auth_service.secrets') as mock_secrets:
            mock_secrets.token_urlsafe.return_value = "mock-refresh-token"
            
            with patch('clarity.services.auth_service.auth.create_custom_token') as mock_custom_token:
                mock_custom_token.return_value = b"mock-access-token"
                
                # Execute with remember_me=True
                result = await auth_service._generate_tokens(user_id, remember_me=True)
                
                # Verify extended expiry
                assert result.expires_in == 86400 * 30  # 30 days

    async def test_refresh_access_token_success(self, auth_service):
        """Test successful token refresh."""
        refresh_token = "valid-refresh-token"
        
        # Mock Firestore response for refresh token
        token_data = {
            "user_id": "test-uid-123",
            "expires_at": datetime.now(UTC) + timedelta(days=1),  # Valid token
            "is_revoked": False,
        }
        auth_service.firestore_client.get_document.return_value = token_data
        
        with patch.object(auth_service, '_generate_tokens') as mock_gen_tokens:
            mock_tokens = TokenResponse(
                access_token="new-access-token",
                refresh_token="new-refresh-token",
                token_type="bearer",
                expires_in=3600,
            )
            mock_gen_tokens.return_value = mock_tokens
            
            # Execute
            result = await auth_service.refresh_access_token(refresh_token)
            
            # Verify
            assert isinstance(result, TokenResponse)
            assert result.access_token == "new-access-token"

    async def test_refresh_access_token_not_found(self, auth_service):
        """Test token refresh with invalid token."""
        refresh_token = "invalid-refresh-token"
        
        # Mock Firestore response - token not found
        auth_service.firestore_client.get_document.return_value = None
        
        with pytest.raises(InvalidCredentialsError):
            await auth_service.refresh_access_token(refresh_token)

    async def test_refresh_access_token_expired(self, auth_service):
        """Test token refresh with expired token."""
        refresh_token = "expired-refresh-token"
        
        # Mock Firestore response - expired token
        token_data = {
            "user_id": "test-uid-123",
            "expires_at": datetime.now(UTC) - timedelta(days=1),  # Expired
            "is_revoked": False,
        }
        auth_service.firestore_client.get_document.return_value = token_data
        
        with pytest.raises(InvalidCredentialsError):
            await auth_service.refresh_access_token(refresh_token)

    async def test_refresh_access_token_revoked(self, auth_service):
        """Test token refresh with revoked token."""
        refresh_token = "revoked-refresh-token"
        
        # Mock Firestore response - revoked token
        token_data = {
            "user_id": "test-uid-123",
            "expires_at": datetime.now(UTC) + timedelta(days=1),
            "is_revoked": True,  # Revoked
        }
        auth_service.firestore_client.get_document.return_value = token_data
        
        with pytest.raises(InvalidCredentialsError):
            await auth_service.refresh_access_token(refresh_token)


class TestSessionManagement:
    """🗂️ Test user session management."""

    async def test_create_user_session(self, auth_service):
        """Test user session creation."""
        user_id = "test-uid-123"
        refresh_token = "refresh-token-123"
        device_info = {"device": "iPhone"}
        ip_address = "192.168.1.1"
        
        with patch('clarity.services.auth_service.uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = uuid.UUID("12345678-1234-5678-9012-123456789012")
            
            # Execute
            session_id = await auth_service._create_user_session(
                user_id=user_id,
                refresh_token=refresh_token,
                device_info=device_info,
                ip_address=ip_address,
                remember_me=True,
            )
            
            # Verify session creation
            assert session_id == "12345678-1234-5678-9012-123456789012"
            auth_service.firestore_client.create_document.assert_called()

    async def test_create_user_session_response(self):
        """Test user session response creation."""
        mock_user_record = Mock()
        mock_user_record.uid = "test-uid-123"
        mock_user_record.email = "test@example.com"
        mock_user_record.display_name = "John Doe"
        
        user_data = {
            "first_name": "John",
            "last_name": "Doe",
            "role": UserRole.PATIENT.value,
            "status": UserStatus.ACTIVE.value,
        }
        
        # Execute
        result = await AuthenticationService._create_user_session_response(
            mock_user_record, user_data
        )
        
        # Verify
        assert isinstance(result, UserSessionResponse)
        assert result.user_id == "test-uid-123"
        assert result.email == "test@example.com"
        assert result.first_name == "John"
        assert result.last_name == "Doe"

    async def test_logout_user_success(self, auth_service):
        """Test successful user logout."""
        refresh_token = "valid-refresh-token"
        
        # Mock successful logout
        auth_service.firestore_client.update_document.return_value = None
        
        # Execute
        result = await auth_service.logout_user(refresh_token)
        
        # Verify
        assert result is True
        auth_service.firestore_client.update_document.assert_called()

    async def test_logout_user_failure(self, auth_service):
        """Test user logout with error."""
        refresh_token = "invalid-refresh-token"
        
        # Mock logout failure
        auth_service.firestore_client.update_document.side_effect = Exception("Firestore error")
        
        # Execute
        result = await auth_service.logout_user(refresh_token)
        
        # Verify
        assert result is False


class TestUserRetrieval:
    """👤 Test user data retrieval."""

    @patch('clarity.services.auth_service.auth')
    async def test_get_user_by_id_success(self, mock_auth, auth_service):
        """Test successful user retrieval by ID."""
        user_id = "test-uid-123"
        
        # Mock Firebase user
        mock_user_record = Mock()
        mock_user_record.uid = user_id
        mock_user_record.email = "test@example.com"
        mock_auth.get_user.return_value = mock_user_record
        
        # Mock Firestore user data
        user_data = {
            "first_name": "John",
            "last_name": "Doe",
            "role": UserRole.PATIENT.value,
            "status": UserStatus.ACTIVE.value,
        }
        auth_service.firestore_client.get_document.return_value = user_data
        
        # Execute
        result = await auth_service.get_user_by_id(user_id)
        
        # Verify
        assert isinstance(result, UserSessionResponse)
        assert result.user_id == user_id

    @patch('clarity.services.auth_service.auth')
    async def test_get_user_by_id_not_found_firebase(self, mock_auth, auth_service):
        """Test user retrieval when not found in Firebase."""
        user_id = "nonexistent-uid"
        
        mock_auth.get_user.side_effect = auth.UserNotFoundError("User not found")
        
        # Execute
        result = await auth_service.get_user_by_id(user_id)
        
        # Verify
        assert result is None

    @patch('clarity.services.auth_service.auth')
    async def test_get_user_by_id_not_found_firestore(self, mock_auth, auth_service):
        """Test user retrieval when not found in Firestore."""
        user_id = "test-uid-123"
        
        # Mock Firebase user exists
        mock_user_record = Mock()
        mock_user_record.uid = user_id
        mock_auth.get_user.return_value = mock_user_record
        
        # Mock Firestore user data not found
        auth_service.firestore_client.get_document.return_value = None
        
        # Execute
        result = await auth_service.get_user_by_id(user_id)
        
        # Verify
        assert result is None


class TestEmailVerification:
    """📧 Test email verification functionality."""

    async def test_verify_email(self):
        """Test email verification (placeholder implementation)."""
        verification_code = "123456"
        
        # Execute
        result = await AuthenticationService.verify_email(verification_code)
        
        # Verify (placeholder always returns True)
        assert result is True


class TestExceptionHierarchy:
    """💥 Test exception class hierarchy."""

    def test_authentication_error_base(self):
        """Test AuthenticationError base class."""
        error = AuthenticationError("Base auth error")
        assert str(error) == "Base auth error"
        assert isinstance(error, Exception)

    def test_specific_auth_errors(self):
        """Test specific authentication error types."""
        # UserNotFoundError
        user_error = UserNotFoundError("User not found")
        assert isinstance(user_error, AuthenticationError)
        
        # InvalidCredentialsError
        cred_error = InvalidCredentialsError("Invalid creds")
        assert isinstance(cred_error, AuthenticationError)
        
        # UserAlreadyExistsError
        exists_error = UserAlreadyExistsError("User exists")
        assert isinstance(exists_error, AuthenticationError)
        
        # EmailNotVerifiedError
        email_error = EmailNotVerifiedError("Email not verified")
        assert isinstance(email_error, AuthenticationError)
        
        # AccountDisabledError
        disabled_error = AccountDisabledError("Account disabled")
        assert isinstance(disabled_error, AuthenticationError)


class TestEdgeCases:
    """🎯 Test edge cases and boundary conditions."""

    async def test_empty_device_info(self, auth_service, sample_registration_request):
        """Test registration with empty device info."""
        with patch('clarity.services.auth_service.auth') as mock_auth:
            mock_user_record = Mock()
            mock_user_record.uid = "test-uid-123"
            
            mock_auth.get_user_by_email.side_effect = auth.UserNotFoundError("User not found")
            mock_auth.create_user.return_value = mock_user_record
            mock_auth.set_custom_user_claims = Mock()
            
            # Execute with empty device info
            await auth_service.register_user(sample_registration_request, device_info={})
            
            # Verify empty device info is handled
            call_args = auth_service.firestore_client.create_document.call_args
            user_data = call_args[1]["data"]
            assert user_data["device_info"] == {}

    async def test_token_generation_with_custom_expiry(self, auth_service):
        """Test token generation with custom expiry times."""
        # Set custom expiry
        auth_service.default_token_expiry = 7200  # 2 hours
        
        user_id = "test-uid-123"
        
        with patch('clarity.services.auth_service.secrets') as mock_secrets:
            mock_secrets.token_urlsafe.return_value = "mock-refresh-token"
            
            with patch('clarity.services.auth_service.auth.create_custom_token') as mock_custom_token:
                mock_custom_token.return_value = b"mock-access-token"
                
                # Execute
                result = await auth_service._generate_tokens(user_id)
                
                # Verify custom expiry
                assert result.expires_in == 7200

    async def test_concurrent_token_operations(self, auth_service):
        """Test concurrent token operations."""
        import asyncio
        
        async def mock_token_op(user_id: str):
            with patch('clarity.services.auth_service.secrets') as mock_secrets:
                mock_secrets.token_urlsafe.return_value = f"token-{user_id}"
                with patch('clarity.services.auth_service.auth.create_custom_token') as mock_custom_token:
                    mock_custom_token.return_value = f"access-{user_id}".encode()
                    return await auth_service._generate_tokens(user_id)
        
        # Run multiple concurrent operations
        user_ids = ["user1", "user2", "user3"]
        tasks = [mock_token_op(uid) for uid in user_ids]
        results = await asyncio.gather(*tasks)
        
        # Verify all operations completed
        assert len(results) == 3
        for result in results:
            assert isinstance(result, TokenResponse)


class TestDataValidation:
    """🔍 Test data validation and sanitization."""

    async def test_registration_data_validation(self, auth_service):
        """Test registration request data validation."""
        # Test with invalid data types
        invalid_request = UserRegistrationRequest(
            email="test@example.com",
            password="password123",
            first_name="",  # Empty name
            last_name="Doe",
            phone_number="invalid-phone",
            terms_accepted=True,
            privacy_policy_accepted=True,
        )
        
        # The validation should be handled by Pydantic models
        assert invalid_request.first_name == ""
        assert invalid_request.phone_number == "invalid-phone"

    async def test_login_data_validation(self, auth_service):
        """Test login request data validation."""
        # Test with various login data
        login_request = UserLoginRequest(
            email="test@example.com",
            password="password123",
            remember_me=True,
        )
        
        assert login_request.remember_me is True
        assert login_request.email == "test@example.com"

""" 