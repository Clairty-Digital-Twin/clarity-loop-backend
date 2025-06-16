"""Tests for Cognito email verification and password reset endpoints."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from fastapi.testclient import TestClient
from botocore.exceptions import ClientError

from clarity.main import app
from clarity.auth.aws_cognito_provider import CognitoAuthProvider


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_cognito_client():
    """Create a mock Cognito client."""
    mock = MagicMock()
    mock.exceptions.CodeMismatchException = type('CodeMismatchException', (Exception,), {})
    mock.exceptions.UserNotFoundException = type('UserNotFoundException', (Exception,), {})
    return mock


class TestEmailConfirmation:
    """Test email confirmation endpoint."""

    @patch("clarity.api.v1.auth.get_auth_provider")
    def test_confirm_email_success(self, mock_get_auth_provider, client, mock_cognito_client):
        """Test successful email confirmation."""
        # Arrange
        mock_provider = MagicMock(spec=CognitoAuthProvider)
        mock_provider.cognito_client = mock_cognito_client
        mock_provider.client_id = "test-client-id"
        mock_get_auth_provider.return_value = mock_provider
        
        # Act
        response = client.post(
            "/api/v1/auth/confirm-email",
            json={"email": "test@example.com", "code": "123456"}
        )
        
        # Assert
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.json()}")
        assert response.status_code == 200
        assert response.json() == {"status": "confirmed"}
        mock_cognito_client.confirm_sign_up.assert_called_once_with(
            ClientId="test-client-id",
            Username="test@example.com",
            ConfirmationCode="123456"
        )

    @patch("clarity.api.v1.auth.get_auth_provider")
    def test_confirm_email_invalid_code(self, mock_get_auth_provider, client, mock_cognito_client):
        """Test email confirmation with invalid code."""
        # Arrange
        mock_provider = MagicMock(spec=CognitoAuthProvider)
        mock_provider.cognito_client = mock_cognito_client
        mock_get_auth_provider.return_value = mock_provider
        
        mock_cognito_client.confirm_sign_up.side_effect = mock_cognito_client.exceptions.CodeMismatchException()
        
        # Act
        response = client.post(
            "/api/v1/auth/confirm-email",
            json={"email": "test@example.com", "code": "wrong"}
        )
        
        # Assert
        assert response.status_code == 400
        assert response.json()["detail"]["type"] == "invalid_code"

    @patch("clarity.api.v1.auth.get_auth_provider")
    def test_confirm_email_user_not_found(self, mock_get_auth_provider, client, mock_cognito_client):
        """Test email confirmation with non-existent user."""
        # Arrange
        mock_provider = MagicMock(spec=CognitoAuthProvider)
        mock_provider.cognito_client = mock_cognito_client
        mock_get_auth_provider.return_value = mock_provider
        
        mock_cognito_client.confirm_sign_up.side_effect = mock_cognito_client.exceptions.UserNotFoundException()
        
        # Act
        response = client.post(
            "/api/v1/auth/confirm-email",
            json={"email": "notfound@example.com", "code": "123456"}
        )
        
        # Assert
        assert response.status_code == 404
        assert response.json()["detail"]["type"] == "user_not_found"


class TestResendConfirmation:
    """Test resend confirmation code endpoint."""

    @patch("clarity.api.v1.auth.get_auth_provider")
    def test_resend_confirmation_success(self, mock_get_auth_provider, client, mock_cognito_client):
        """Test successful resend of confirmation code."""
        # Arrange
        mock_provider = MagicMock(spec=CognitoAuthProvider)
        mock_provider.cognito_client = mock_cognito_client
        mock_provider.client_id = "test-client-id"
        mock_get_auth_provider.return_value = mock_provider
        
        # Act
        response = client.post(
            "/api/v1/auth/resend-confirmation",
            json={"email": "test@example.com"}
        )
        
        # Assert
        assert response.status_code == 200
        assert response.json() == {"status": "sent"}
        mock_cognito_client.resend_confirmation_code.assert_called_once_with(
            ClientId="test-client-id",
            Username="test@example.com"
        )

    @patch("clarity.api.v1.auth.get_auth_provider")
    def test_resend_confirmation_user_not_found(self, mock_get_auth_provider, client, mock_cognito_client):
        """Test resend confirmation for non-existent user."""
        # Arrange
        mock_provider = MagicMock(spec=CognitoAuthProvider)
        mock_provider.cognito_client = mock_cognito_client
        mock_get_auth_provider.return_value = mock_provider
        
        mock_cognito_client.resend_confirmation_code.side_effect = mock_cognito_client.exceptions.UserNotFoundException()
        
        # Act
        response = client.post(
            "/api/v1/auth/resend-confirmation",
            json={"email": "notfound@example.com"}
        )
        
        # Assert
        assert response.status_code == 404
        assert response.json()["detail"]["type"] == "user_not_found"


class TestForgotPassword:
    """Test forgot password endpoint."""

    @patch("clarity.api.v1.auth.get_auth_provider")
    def test_forgot_password_success(self, mock_get_auth_provider, client, mock_cognito_client):
        """Test successful forgot password request."""
        # Arrange
        mock_provider = MagicMock(spec=CognitoAuthProvider)
        mock_provider.cognito_client = mock_cognito_client
        mock_provider.client_id = "test-client-id"
        mock_get_auth_provider.return_value = mock_provider
        
        # Act
        response = client.post(
            "/api/v1/auth/forgot-password",
            json={"email": "test@example.com"}
        )
        
        # Assert
        assert response.status_code == 200
        assert response.json() == {"status": "sent"}
        mock_cognito_client.forgot_password.assert_called_once_with(
            ClientId="test-client-id",
            Username="test@example.com"
        )

    @patch("clarity.api.v1.auth.get_auth_provider")
    def test_forgot_password_user_not_found(self, mock_get_auth_provider, client, mock_cognito_client):
        """Test forgot password for non-existent user."""
        # Arrange
        mock_provider = MagicMock(spec=CognitoAuthProvider)
        mock_provider.cognito_client = mock_cognito_client
        mock_get_auth_provider.return_value = mock_provider
        
        mock_cognito_client.forgot_password.side_effect = mock_cognito_client.exceptions.UserNotFoundException()
        
        # Act
        response = client.post(
            "/api/v1/auth/forgot-password",
            json={"email": "notfound@example.com"}
        )
        
        # Assert
        assert response.status_code == 404
        assert response.json()["detail"]["type"] == "user_not_found"


class TestResetPassword:
    """Test reset password endpoint."""

    @patch("clarity.api.v1.auth.get_auth_provider")
    def test_reset_password_success(self, mock_get_auth_provider, client, mock_cognito_client):
        """Test successful password reset."""
        # Arrange
        mock_provider = MagicMock(spec=CognitoAuthProvider)
        mock_provider.cognito_client = mock_cognito_client
        mock_provider.client_id = "test-client-id"
        mock_get_auth_provider.return_value = mock_provider
        
        # Act
        response = client.post(
            "/api/v1/auth/reset-password",
            json={
                "email": "test@example.com",
                "code": "123456",
                "new_password": "NewPassword123!"
            }
        )
        
        # Assert
        assert response.status_code == 200
        assert response.json() == {"status": "reset"}
        mock_cognito_client.confirm_forgot_password.assert_called_once_with(
            ClientId="test-client-id",
            Username="test@example.com",
            ConfirmationCode="123456",
            Password="NewPassword123!"
        )

    @patch("clarity.api.v1.auth.get_auth_provider")
    def test_reset_password_invalid_code(self, mock_get_auth_provider, client, mock_cognito_client):
        """Test password reset with invalid code."""
        # Arrange
        mock_provider = MagicMock(spec=CognitoAuthProvider)
        mock_provider.cognito_client = mock_cognito_client
        mock_get_auth_provider.return_value = mock_provider
        
        mock_cognito_client.confirm_forgot_password.side_effect = mock_cognito_client.exceptions.CodeMismatchException()
        
        # Act
        response = client.post(
            "/api/v1/auth/reset-password",
            json={
                "email": "test@example.com",
                "code": "wrong",
                "new_password": "NewPassword123!"
            }
        )
        
        # Assert
        assert response.status_code == 400
        assert response.json()["detail"]["type"] == "invalid_code"

    @patch("clarity.api.v1.auth.get_auth_provider")
    def test_reset_password_user_not_found(self, mock_get_auth_provider, client, mock_cognito_client):
        """Test password reset for non-existent user."""
        # Arrange
        mock_provider = MagicMock(spec=CognitoAuthProvider)
        mock_provider.cognito_client = mock_cognito_client
        mock_get_auth_provider.return_value = mock_provider
        
        mock_cognito_client.confirm_forgot_password.side_effect = mock_cognito_client.exceptions.UserNotFoundException()
        
        # Act
        response = client.post(
            "/api/v1/auth/reset-password",
            json={
                "email": "notfound@example.com",
                "code": "123456",
                "new_password": "NewPassword123!"
            }
        )
        
        # Assert
        assert response.status_code == 404
        assert response.json()["detail"]["type"] == "user_not_found"

    def test_reset_password_weak_password(self, client):
        """Test password reset with weak password."""
        # Act
        response = client.post(
            "/api/v1/auth/reset-password",
            json={
                "email": "test@example.com",
                "code": "123456",
                "new_password": "weak"
            }
        )
        
        # Assert
        assert response.status_code == 422  # Validation error
        assert "min_length" in str(response.json())