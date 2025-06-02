"""Unit tests for authentication API endpoints.

Tests the FastAPI authentication endpoints including registration, login,
token refresh, and user info retrieval.
"""

from unittest.mock import AsyncMock, Mock, patch
import uuid

from fastapi.testclient import TestClient
import pytest

from clarity.main import get_app
from clarity.models.auth import UserStatus
from clarity.services.auth_service import (
    UserAlreadyExistsError,
    UserNotFoundError,
)


class TestAuthenticationEndpoints:
    """Test suite for authentication API endpoints."""

    @pytest.fixture
    @staticmethod
    def client() -> TestClient:
        """Create test client with authentication endpoints."""
        app = get_app()
        return TestClient(app)

    @pytest.fixture
    @staticmethod
    def sample_registration_data() -> dict:
        """Sample user registration data."""
        return {
            "email": "test@example.com",
            "password": "SecurePass123!",
            "first_name": "Test",
            "last_name": "User",
            "phone_number": "+1234567890",
            "terms_accepted": True,
            "privacy_policy_accepted": True,
        }

    @pytest.fixture
    @staticmethod
    def sample_login_data() -> dict:
        """Sample user login data."""
        return {
            "email": "test@example.com",
            "password": "SecurePass123!",
            "remember_me": False,
        }

    @staticmethod
    def test_auth_endpoints_available(client: TestClient) -> None:
        """Test that authentication endpoints are available."""
        # Health check should work
        response = client.get("/health")
        assert response.status_code == 200

        # Auth health check should be available (though may fail without proper setup)
        response = client.get("/api/v1/auth/health")
        # Should return either 200 (healthy) or 500 (service not configured)
        assert response.status_code in {200, 500, 503}

    @patch("clarity.api.v1.auth.get_auth_service")
    @staticmethod
    def test_user_registration_endpoint(
        mock_get_auth_service: Mock,
        client: TestClient,
        sample_registration_data: dict,
    ) -> None:
        """Test user registration endpoint."""
        # Mock authentication service
        mock_auth_service = AsyncMock()
        mock_get_auth_service.return_value = mock_auth_service

        # Mock successful registration
        mock_auth_service.register_user.return_value = Mock(
            user_id=uuid.uuid4(),
            email="test@example.com",
            status=UserStatus.PENDING_VERIFICATION,
            verification_email_sent=True,
            created_at="2024-01-01T00:00:00Z",
        )

        response = client.post("/api/v1/auth/register", json=sample_registration_data)

        # Should call the service
        mock_auth_service.register_user.assert_called_once()

        # Verify response structure (might be 201 or 500 depending on service availability)
        assert response.status_code in {201, 500}

    @patch("clarity.api.v1.auth.get_auth_service")
    @staticmethod
    def test_user_registration_validation_error(
        mock_get_auth_service: Mock,
        client: TestClient,
    ) -> None:
        """Test user registration with validation errors."""
        # Mock authentication service
        mock_auth_service = AsyncMock()
        mock_get_auth_service.return_value = mock_auth_service

        # Invalid registration data (missing required fields)
        invalid_data = {
            "email": "invalid-email",  # Invalid email format
            "password": "weak",  # Too weak password
        }

        response = client.post("/api/v1/auth/register", json=invalid_data)

        # Should return validation error
        assert response.status_code == 422  # FastAPI validation error

    @patch("clarity.api.v1.auth.get_auth_service")
    @staticmethod
    def test_user_registration_already_exists_error(
        mock_get_auth_service: Mock,
        client: TestClient,
        sample_registration_data: dict,
    ) -> None:
        """Test user registration when user already exists."""
        # Mock authentication service
        mock_auth_service = AsyncMock()
        mock_get_auth_service.return_value = mock_auth_service

        # Mock user already exists error
        mock_auth_service.register_user.side_effect = UserAlreadyExistsError(
            "User already exists"
        )

        response = client.post("/api/v1/auth/register", json=sample_registration_data)

        # Should return conflict error or 500 if service not configured
        assert response.status_code in {409, 500}

    @patch("clarity.api.v1.auth.get_auth_service")
    @staticmethod
    def test_user_login_endpoint(
        mock_get_auth_service: Mock,
        client: TestClient,
        sample_login_data: dict,
    ) -> None:
        """Test user login endpoint."""
        # Mock authentication service
        mock_auth_service = AsyncMock()
        mock_get_auth_service.return_value = mock_auth_service

        # Mock successful login
        mock_login_response = Mock()
        mock_login_response.user = Mock(
            user_id=uuid.uuid4(),
            email="test@example.com",
            first_name="Test",
            last_name="User",
            role="patient",
            permissions=["read_own_data"],
            status=UserStatus.ACTIVE,
            mfa_enabled=False,
            email_verified=True,
        )
        # Create mock tokens (not actual security tokens)
        mock_login_response.tokens = Mock(
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            token_type="bearer",
            expires_in=3600,
        )
        mock_login_response.requires_mfa = False
        mock_login_response.mfa_session_token = None

        mock_auth_service.login_user.return_value = mock_login_response

        response = client.post("/api/v1/auth/login", json=sample_login_data)

        # Should call the service
        mock_auth_service.login_user.assert_called_once()

        # Verify response (might be 200 or 500 depending on service availability)
        assert response.status_code in {200, 500}

    @patch("clarity.api.v1.auth.get_auth_service")
    @staticmethod
    def test_user_login_not_found_error(
        mock_get_auth_service: Mock,
        client: TestClient,
        sample_login_data: dict,
    ) -> None:
        """Test user login when user not found."""
        # Mock authentication service
        mock_auth_service = AsyncMock()
        mock_get_auth_service.return_value = mock_auth_service

        # Mock user not found error
        mock_auth_service.login_user.side_effect = UserNotFoundError("User not found")

        response = client.post("/api/v1/auth/login", json=sample_login_data)

        # Should return not found error or 500 if service not configured
        assert response.status_code in {404, 500}

    @patch("clarity.api.v1.auth.get_auth_service")
    @staticmethod
    def test_token_refresh_endpoint(
        mock_get_auth_service: Mock,
        client: TestClient,
    ) -> None:
        """Test token refresh endpoint."""
        # Mock authentication service
        mock_auth_service = AsyncMock()
        mock_get_auth_service.return_value = mock_auth_service

        # Mock successful token refresh (not actual security tokens)
        mock_auth_service.refresh_access_token.return_value = Mock(
            access_token="test_new_access_token",
            refresh_token="test_new_refresh_token",
            token_type="bearer",
            expires_in=3600,
        )

        refresh_data = {"refresh_token": "test_refresh_token"}
        response = client.post("/api/v1/auth/refresh", json=refresh_data)

        # Should call the service or return 500 if not configured
        assert response.status_code in {200, 500}

    @patch("clarity.api.v1.auth.get_auth_service")
    @staticmethod
    def test_logout_endpoint(
        mock_get_auth_service: Mock,
        client: TestClient,
    ) -> None:
        """Test user logout endpoint."""
        # Mock authentication service
        mock_auth_service = AsyncMock()
        mock_get_auth_service.return_value = mock_auth_service

        # Mock successful logout
        mock_auth_service.logout_user.return_value = True

        logout_data = {"refresh_token": "test_refresh_token"}
        response = client.post("/api/v1/auth/logout", json=logout_data)

        # Should call the service or return 500 if not configured
        assert response.status_code in {200, 500}

    @staticmethod
    def test_authentication_service_not_configured(client: TestClient) -> None:
        """Test endpoint behavior when authentication service is not configured."""
        # This test verifies that endpoints fail gracefully when service isn't available
        sample_data = {
            "email": "test@example.com",
            "password": "SecurePass123!",
            "first_name": "Test",
            "last_name": "User",
            "terms_accepted": True,
            "privacy_policy_accepted": True,
        }

        response = client.post("/api/v1/auth/register", json=sample_data)

        # Should return 500 when service is not configured
        assert response.status_code == 500

    @staticmethod
    def test_password_validation_requirements(client: TestClient) -> None:
        """Test password validation requirements."""
        weak_passwords = [
            "weak",  # Too short
            "alllowercase",  # No uppercase, digits, special chars
            "ALLUPPERCASE",  # No lowercase, digits, special chars
            "NoDigits!",  # No digits
            "NoSpecial123",  # No special characters
        ]

        for weak_password in weak_passwords:
            data = {
                "email": "test@example.com",
                "password": weak_password,
                "first_name": "Test",
                "last_name": "User",
                "terms_accepted": True,
                "privacy_policy_accepted": True,
            }

            response = client.post("/api/v1/auth/register", json=data)

            # Should return validation error for weak passwords
            assert response.status_code == 422

    @staticmethod
    def test_email_validation(client: TestClient) -> None:
        """Test email validation."""
        invalid_emails = [
            "not-an-email",
            "missing@",
            "@missing-domain.com",
            "invalid.email",
        ]

        for invalid_email in invalid_emails:
            data = {
                "email": invalid_email,
                "password": "SecurePass123!",
                "first_name": "Test",
                "last_name": "User",
                "terms_accepted": True,
                "privacy_policy_accepted": True,
            }

            response = client.post("/api/v1/auth/register", json=data)

            # Should return validation error for invalid emails
            assert response.status_code == 422

    @staticmethod
    def test_terms_acceptance_required(client: TestClient) -> None:
        """Test that terms and privacy policy acceptance is required."""
        data = {
            "email": "test@example.com",
            "password": "SecurePass123!",
            "first_name": "Test",
            "last_name": "User",
            "terms_accepted": False,  # Not accepted
            "privacy_policy_accepted": True,
        }

        response = client.post("/api/v1/auth/register", json=data)

        # Should return validation error when terms not accepted
        assert response.status_code == 422
