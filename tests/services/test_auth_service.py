"""Comprehensive tests for AuthService functionality.

Tests cover:
- User registration with validation
- User authentication and login
- Token management (creation, refresh, validation)
- Password validation and security
- Error handling and edge cases
- Integration with Firebase Auth
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from clarity.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ValidationError,
)
from clarity.models.auth import (
    AuthTokens,
    LoginRequest,
    RegisterRequest,
    TokenRefreshRequest,
    UserProfile,
    UserRole,
)
from clarity.services.auth_service import AuthService


@pytest.fixture
def mock_firebase_auth():
    """Mock Firebase Auth provider."""
    mock_auth = AsyncMock()
    return mock_auth


@pytest.fixture
def auth_service(mock_firebase_auth):
    """Create AuthService instance with mocked dependencies."""
    return AuthService(auth_provider=mock_firebase_auth)


@pytest.fixture
def sample_register_request():
    """Sample registration request."""
    return RegisterRequest(
        email="test@example.com",
        password="SecurePass123!",
        confirm_password="SecurePass123!",
        first_name="John",
        last_name="Doe",
        role=UserRole.PATIENT,
        terms_accepted=True,
    )


@pytest.fixture
def sample_login_request():
    """Sample login request."""
    return LoginRequest(
        email="test@example.com",
        password="SecurePass123!",
    )


@pytest.fixture
def sample_user_profile():
    """Sample user profile."""
    return UserProfile(
        user_id=uuid4(),
        email="test@example.com",
        first_name="John",
        last_name="Doe",
        role=UserRole.PATIENT,
        is_active=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


@pytest.fixture
def sample_auth_tokens():
    """Sample auth tokens."""
    return AuthTokens(
        access_token="access_token_123",
        refresh_token="refresh_token_456",
        token_type="Bearer",
        expires_in=3600,
        user_id=str(uuid4()),
    )


class TestAuthServiceInitialization:
    """Test AuthService initialization."""

    def test_init_with_auth_provider(self, mock_firebase_auth):
        """Test initialization with auth provider."""
        service = AuthService(auth_provider=mock_firebase_auth)
        assert service.auth_provider is mock_firebase_auth

    def test_init_without_auth_provider(self):
        """Test initialization without auth provider raises error."""
        with pytest.raises(ValueError, match="Auth provider is required"):
            AuthService(auth_provider=None)


class TestAuthServiceRegistration:
    """Test user registration functionality."""

    @pytest.mark.asyncio
    async def test_register_user_success(self, auth_service, sample_register_request, sample_user_profile):
        """Test successful user registration."""
        # Mock Firebase Auth response
        auth_service.auth_provider.create_user.return_value = {
            "uid": str(sample_user_profile.user_id),
            "email": sample_register_request.email,
        }
        auth_service.auth_provider.get_user_profile.return_value = sample_user_profile

        result = await auth_service.register_user(sample_register_request)

        assert result.email == sample_register_request.email
        assert result.first_name == sample_register_request.first_name
        auth_service.auth_provider.create_user.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_user_password_mismatch(self, auth_service):
        """Test registration with password mismatch."""
        request = RegisterRequest(
            email="test@example.com",
            password="SecurePass123!",
            confirm_password="DifferentPass123!",
            first_name="John",
            last_name="Doe",
            role=UserRole.PATIENT,
            terms_accepted=True,
        )

        with pytest.raises(ValidationError, match="Passwords do not match"):
            await auth_service.register_user(request)

    @pytest.mark.asyncio
    async def test_register_user_weak_password(self, auth_service):
        """Test registration with weak password."""
        request = RegisterRequest(
            email="test@example.com",
            password="weak",
            confirm_password="weak",
            first_name="John",
            last_name="Doe",
            role=UserRole.PATIENT,
            terms_accepted=True,
        )

        with pytest.raises(ValidationError, match="Password does not meet security requirements"):
            await auth_service.register_user(request)

    @pytest.mark.asyncio
    async def test_register_user_terms_not_accepted(self, auth_service):
        """Test registration without accepting terms."""
        request = RegisterRequest(
            email="test@example.com",
            password="SecurePass123!",
            confirm_password="SecurePass123!",
            first_name="John",
            last_name="Doe",
            role=UserRole.PATIENT,
            terms_accepted=False,
        )

        with pytest.raises(ValidationError, match="Terms and conditions must be accepted"):
            await auth_service.register_user(request)

    @pytest.mark.asyncio
    async def test_register_user_invalid_email(self, auth_service):
        """Test registration with invalid email."""
        request = RegisterRequest(
            email="invalid-email",
            password="SecurePass123!",
            confirm_password="SecurePass123!",
            first_name="John",
            last_name="Doe",
            role=UserRole.PATIENT,
            terms_accepted=True,
        )

        with pytest.raises(ValidationError, match="Invalid email format"):
            await auth_service.register_user(request)

    @pytest.mark.asyncio
    async def test_register_user_already_exists(self, auth_service, sample_register_request):
        """Test registration when user already exists."""
        auth_service.auth_provider.create_user.side_effect = AuthenticationError("Email already in use")

        with pytest.raises(AuthenticationError, match="Email already in use"):
            await auth_service.register_user(sample_register_request)

    @pytest.mark.asyncio
    async def test_register_user_firebase_error(self, auth_service, sample_register_request):
        """Test registration with Firebase error."""
        auth_service.auth_provider.create_user.side_effect = Exception("Firebase error")

        with pytest.raises(AuthenticationError, match="Registration failed"):
            await auth_service.register_user(sample_register_request)


class TestAuthServiceLogin:
    """Test user login functionality."""

    @pytest.mark.asyncio
    async def test_login_user_success(self, auth_service, sample_login_request, sample_auth_tokens):
        """Test successful user login."""
        auth_service.auth_provider.authenticate_user.return_value = sample_auth_tokens

        result = await auth_service.login_user(sample_login_request)

        assert result.access_token == sample_auth_tokens.access_token
        assert result.user_id == sample_auth_tokens.user_id
        auth_service.auth_provider.authenticate_user.assert_called_once_with(
            sample_login_request.email, sample_login_request.password
        )

    @pytest.mark.asyncio
    async def test_login_user_invalid_credentials(self, auth_service, sample_login_request):
        """Test login with invalid credentials."""
        auth_service.auth_provider.authenticate_user.side_effect = AuthenticationError("Invalid credentials")

        with pytest.raises(AuthenticationError, match="Invalid credentials"):
            await auth_service.login_user(sample_login_request)

    @pytest.mark.asyncio
    async def test_login_user_disabled_account(self, auth_service, sample_login_request):
        """Test login with disabled account."""
        auth_service.auth_provider.authenticate_user.side_effect = AuthenticationError("Account disabled")

        with pytest.raises(AuthenticationError, match="Account disabled"):
            await auth_service.login_user(sample_login_request)

    @pytest.mark.asyncio
    async def test_login_user_firebase_error(self, auth_service, sample_login_request):
        """Test login with Firebase error."""
        auth_service.auth_provider.authenticate_user.side_effect = Exception("Firebase error")

        with pytest.raises(AuthenticationError, match="Authentication failed"):
            await auth_service.login_user(sample_login_request)


class TestAuthServiceTokenManagement:
    """Test token management functionality."""

    @pytest.mark.asyncio
    async def test_refresh_token_success(self, auth_service, sample_auth_tokens):
        """Test successful token refresh."""
        refresh_request = TokenRefreshRequest(refresh_token="refresh_token_456")
        auth_service.auth_provider.refresh_token.return_value = sample_auth_tokens

        result = await auth_service.refresh_token(refresh_request)

        assert result.access_token == sample_auth_tokens.access_token
        auth_service.auth_provider.refresh_token.assert_called_once_with("refresh_token_456")

    @pytest.mark.asyncio
    async def test_refresh_token_invalid(self, auth_service):
        """Test token refresh with invalid token."""
        refresh_request = TokenRefreshRequest(refresh_token="invalid_token")
        auth_service.auth_provider.refresh_token.side_effect = AuthenticationError("Invalid refresh token")

        with pytest.raises(AuthenticationError, match="Invalid refresh token"):
            await auth_service.refresh_token(refresh_request)

    @pytest.mark.asyncio
    async def test_refresh_token_expired(self, auth_service):
        """Test token refresh with expired token."""
        refresh_request = TokenRefreshRequest(refresh_token="expired_token")
        auth_service.auth_provider.refresh_token.side_effect = AuthenticationError("Refresh token expired")

        with pytest.raises(AuthenticationError, match="Refresh token expired"):
            await auth_service.refresh_token(refresh_request)

    @pytest.mark.asyncio
    async def test_validate_token_success(self, auth_service, sample_user_profile):
        """Test successful token validation."""
        token = "valid_token"
        auth_service.auth_provider.verify_token.return_value = {
            "uid": str(sample_user_profile.user_id),
            "email": sample_user_profile.email,
        }
        auth_service.auth_provider.get_user_profile.return_value = sample_user_profile

        result = await auth_service.validate_token(token)

        assert result.user_id == sample_user_profile.user_id
        auth_service.auth_provider.verify_token.assert_called_once_with(token)

    @pytest.mark.asyncio
    async def test_validate_token_invalid(self, auth_service):
        """Test token validation with invalid token."""
        token = "invalid_token"
        auth_service.auth_provider.verify_token.side_effect = AuthenticationError("Invalid token")

        with pytest.raises(AuthenticationError, match="Invalid token"):
            await auth_service.validate_token(token)

    @pytest.mark.asyncio
    async def test_validate_token_expired(self, auth_service):
        """Test token validation with expired token."""
        token = "expired_token"
        auth_service.auth_provider.verify_token.side_effect = AuthenticationError("Token expired")

        with pytest.raises(AuthenticationError, match="Token expired"):
            await auth_service.validate_token(token)

    @pytest.mark.asyncio
    async def test_logout_user_success(self, auth_service):
        """Test successful user logout."""
        user_id = str(uuid4())
        auth_service.auth_provider.revoke_tokens.return_value = True

        result = await auth_service.logout_user(user_id)

        assert result is True
        auth_service.auth_provider.revoke_tokens.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_logout_user_error(self, auth_service):
        """Test logout with error."""
        user_id = str(uuid4())
        auth_service.auth_provider.revoke_tokens.side_effect = Exception("Logout failed")

        with pytest.raises(AuthenticationError, match="Logout failed"):
            await auth_service.logout_user(user_id)


class TestAuthServicePasswordValidation:
    """Test password validation functionality."""

    def test_validate_password_strength_valid(self, auth_service):
        """Test password strength validation with valid passwords."""
        valid_passwords = [
            "SecurePass123!",
            "MyStr0ng_P@ssw0rd",
            "C0mpl3x!P@55w0rd",
        ]

        for password in valid_passwords:
            assert auth_service._validate_password_strength(password) is True

    def test_validate_password_strength_too_short(self, auth_service):
        """Test password strength validation with short password."""
        short_password = "Short1!"
        assert auth_service._validate_password_strength(short_password) is False

    def test_validate_password_strength_no_uppercase(self, auth_service):
        """Test password strength validation without uppercase."""
        password = "nouppercase123!"
        assert auth_service._validate_password_strength(password) is False

    def test_validate_password_strength_no_lowercase(self, auth_service):
        """Test password strength validation without lowercase."""
        password = "NOLOWERCASE123!"
        assert auth_service._validate_password_strength(password) is False

    def test_validate_password_strength_no_digit(self, auth_service):
        """Test password strength validation without digit."""
        password = "NoDigits!"
        assert auth_service._validate_password_strength(password) is False

    def test_validate_password_strength_no_special(self, auth_service):
        """Test password strength validation without special character."""
        password = "NoSpecialChars123"
        assert auth_service._validate_password_strength(password) is False

    def test_validate_email_format_valid(self, auth_service):
        """Test email format validation with valid emails."""
        valid_emails = [
            "test@example.com",
            "user.name@domain.org",
            "firstname+lastname@company.co.uk",
        ]

        for email in valid_emails:
            assert auth_service._validate_email_format(email) is True

    def test_validate_email_format_invalid(self, auth_service):
        """Test email format validation with invalid emails."""
        invalid_emails = [
            "invalid.email",
            "@domain.com",
            "user@",
            "user name@domain.com",
        ]

        for email in invalid_emails:
            assert auth_service._validate_email_format(email) is False


class TestAuthServiceUserProfile:
    """Test user profile management."""

    @pytest.mark.asyncio
    async def test_get_user_profile_success(self, auth_service, sample_user_profile):
        """Test successful user profile retrieval."""
        user_id = str(sample_user_profile.user_id)
        auth_service.auth_provider.get_user_profile.return_value = sample_user_profile

        result = await auth_service.get_user_profile(user_id)

        assert result.user_id == sample_user_profile.user_id
        assert result.email == sample_user_profile.email
        auth_service.auth_provider.get_user_profile.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_get_user_profile_not_found(self, auth_service):
        """Test user profile retrieval when user not found."""
        user_id = str(uuid4())
        auth_service.auth_provider.get_user_profile.side_effect = AuthenticationError("User not found")

        with pytest.raises(AuthenticationError, match="User not found"):
            await auth_service.get_user_profile(user_id)

    @pytest.mark.asyncio
    async def test_update_user_profile_success(self, auth_service, sample_user_profile):
        """Test successful user profile update."""
        user_id = str(sample_user_profile.user_id)
        updates = {"first_name": "Jane", "last_name": "Smith"}

        updated_profile = sample_user_profile.model_copy(update=updates)
        auth_service.auth_provider.update_user_profile.return_value = updated_profile

        result = await auth_service.update_user_profile(user_id, updates)

        assert result.first_name == "Jane"
        assert result.last_name == "Smith"
        auth_service.auth_provider.update_user_profile.assert_called_once_with(user_id, updates)

    @pytest.mark.asyncio
    async def test_update_user_profile_validation_error(self, auth_service):
        """Test user profile update with validation error."""
        user_id = str(uuid4())
        invalid_updates = {"email": "invalid-email"}

        with pytest.raises(ValidationError, match="Invalid email format"):
            await auth_service.update_user_profile(user_id, invalid_updates)

    @pytest.mark.asyncio
    async def test_change_password_success(self, auth_service):
        """Test successful password change."""
        user_id = str(uuid4())
        old_password = "OldPass123!"
        new_password = "NewPass123!"

        auth_service.auth_provider.change_password.return_value = True

        result = await auth_service.change_password(user_id, old_password, new_password)

        assert result is True
        auth_service.auth_provider.change_password.assert_called_once_with(
            user_id, old_password, new_password
        )

    @pytest.mark.asyncio
    async def test_change_password_weak_new_password(self, auth_service):
        """Test password change with weak new password."""
        user_id = str(uuid4())
        old_password = "OldPass123!"
        weak_password = "weak"

        with pytest.raises(ValidationError, match="Password does not meet security requirements"):
            await auth_service.change_password(user_id, old_password, weak_password)

    @pytest.mark.asyncio
    async def test_change_password_wrong_old_password(self, auth_service):
        """Test password change with wrong old password."""
        user_id = str(uuid4())
        old_password = "WrongPass123!"
        new_password = "NewPass123!"

        auth_service.auth_provider.change_password.side_effect = AuthenticationError("Invalid current password")

        with pytest.raises(AuthenticationError, match="Invalid current password"):
            await auth_service.change_password(user_id, old_password, new_password)


class TestAuthServiceRoleManagement:
    """Test role-based access control."""

    @pytest.mark.asyncio
    async def test_check_permission_success(self, auth_service, sample_user_profile):
        """Test successful permission check."""
        user_id = str(sample_user_profile.user_id)
        permission = "read_health_data"

        auth_service.auth_provider.get_user_profile.return_value = sample_user_profile
        auth_service.auth_provider.check_permission.return_value = True

        result = await auth_service.check_permission(user_id, permission)

        assert result is True
        auth_service.auth_provider.check_permission.assert_called_once_with(
            sample_user_profile.role, permission
        )

    @pytest.mark.asyncio
    async def test_check_permission_denied(self, auth_service, sample_user_profile):
        """Test permission check denied."""
        user_id = str(sample_user_profile.user_id)
        permission = "admin_access"

        auth_service.auth_provider.get_user_profile.return_value = sample_user_profile
        auth_service.auth_provider.check_permission.return_value = False

        result = await auth_service.check_permission(user_id, permission)

        assert result is False

    @pytest.mark.asyncio
    async def test_require_permission_success(self, auth_service, sample_user_profile):
        """Test successful permission requirement."""
        user_id = str(sample_user_profile.user_id)
        permission = "read_health_data"

        auth_service.auth_provider.get_user_profile.return_value = sample_user_profile
        auth_service.auth_provider.check_permission.return_value = True

        # Should not raise
        await auth_service.require_permission(user_id, permission)

    @pytest.mark.asyncio
    async def test_require_permission_denied(self, auth_service, sample_user_profile):
        """Test permission requirement denied."""
        user_id = str(sample_user_profile.user_id)
        permission = "admin_access"

        auth_service.auth_provider.get_user_profile.return_value = sample_user_profile
        auth_service.auth_provider.check_permission.return_value = False

        with pytest.raises(AuthorizationError, match="Insufficient permissions"):
            await auth_service.require_permission(user_id, permission)


class TestAuthServicePasswordReset:
    """Test password reset functionality."""

    @pytest.mark.asyncio
    async def test_request_password_reset_success(self, auth_service):
        """Test successful password reset request."""
        email = "test@example.com"
        auth_service.auth_provider.send_password_reset_email.return_value = True

        result = await auth_service.request_password_reset(email)

        assert result is True
        auth_service.auth_provider.send_password_reset_email.assert_called_once_with(email)

    @pytest.mark.asyncio
    async def test_request_password_reset_invalid_email(self, auth_service):
        """Test password reset request with invalid email."""
        email = "invalid-email"

        with pytest.raises(ValidationError, match="Invalid email format"):
            await auth_service.request_password_reset(email)

    @pytest.mark.asyncio
    async def test_request_password_reset_user_not_found(self, auth_service):
        """Test password reset request for non-existent user."""
        email = "nonexistent@example.com"
        auth_service.auth_provider.send_password_reset_email.side_effect = AuthenticationError("User not found")

        with pytest.raises(AuthenticationError, match="User not found"):
            await auth_service.request_password_reset(email)

    @pytest.mark.asyncio
    async def test_confirm_password_reset_success(self, auth_service):
        """Test successful password reset confirmation."""
        reset_token = "reset_token_123"
        new_password = "NewPass123!"

        auth_service.auth_provider.confirm_password_reset.return_value = True

        result = await auth_service.confirm_password_reset(reset_token, new_password)

        assert result is True
        auth_service.auth_provider.confirm_password_reset.assert_called_once_with(
            reset_token, new_password
        )

    @pytest.mark.asyncio
    async def test_confirm_password_reset_weak_password(self, auth_service):
        """Test password reset confirmation with weak password."""
        reset_token = "reset_token_123"
        weak_password = "weak"

        with pytest.raises(ValidationError, match="Password does not meet security requirements"):
            await auth_service.confirm_password_reset(reset_token, weak_password)

    @pytest.mark.asyncio
    async def test_confirm_password_reset_invalid_token(self, auth_service):
        """Test password reset confirmation with invalid token."""
        reset_token = "invalid_token"
        new_password = "NewPass123!"

        auth_service.auth_provider.confirm_password_reset.side_effect = AuthenticationError("Invalid reset token")

        with pytest.raises(AuthenticationError, match="Invalid reset token"):
            await auth_service.confirm_password_reset(reset_token, new_password)


class TestAuthServiceErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_service_handles_concurrent_requests(self, auth_service, sample_login_request, sample_auth_tokens):
        """Test service handles concurrent login requests."""
        auth_service.auth_provider.authenticate_user.return_value = sample_auth_tokens

        # Start multiple concurrent login requests
        tasks = [auth_service.login_user(sample_login_request) for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        assert all(isinstance(result, AuthTokens) for result in results)

    @pytest.mark.asyncio
    async def test_service_handles_provider_downtime(self, auth_service, sample_login_request):
        """Test service handles auth provider downtime."""
        auth_service.auth_provider.authenticate_user.side_effect = Exception("Service unavailable")

        with pytest.raises(AuthenticationError, match="Authentication failed"):
            await auth_service.login_user(sample_login_request)

    @pytest.mark.asyncio
    async def test_service_validates_user_id_format(self, auth_service):
        """Test service validates user ID format."""
        invalid_user_id = "invalid-uuid"

        with pytest.raises(ValidationError, match="Invalid user ID format"):
            await auth_service.get_user_profile(invalid_user_id)

    def test_service_handles_none_auth_provider(self):
        """Test service handles None auth provider gracefully."""
        with pytest.raises(ValueError, match="Auth provider is required"):
            AuthService(auth_provider=None)
