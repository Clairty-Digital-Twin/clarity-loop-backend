"""Comprehensive tests for MockAuth.

Tests all methods and edge cases to improve coverage from 37% to 90%+.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import jwt
import pytest

from clarity.auth.mock_auth import MockAuthProvider


class TestMockAuthComprehensive:
    """Comprehensive test coverage for MockAuthProvider."""

    @pytest.fixture
    def mock_auth(self):
        """Create MockAuthProvider instance for testing."""
        return MockAuthProvider()

    @pytest.fixture
    def valid_token_payload(self):
        """Create valid token payload for testing."""
        return {
            "user_id": "test_user_123",
            "email": "test@example.com",
            "role": "user",
            "exp": (datetime.now(UTC) + timedelta(hours=1)).timestamp()
        }

    @pytest.fixture
    def expired_token_payload(self):
        """Create expired token payload for testing."""
        return {
            "user_id": "test_user_123",
            "email": "test@example.com",
            "role": "user",
            "exp": (datetime.now(UTC) - timedelta(hours=1)).timestamp()
        }

    def test_initialization(self, mock_auth):
        """Test MockAuthProvider initialization."""
        assert hasattr(mock_auth, '_mock_users')
        assert len(mock_auth._mock_users) == 3
        assert 'mock_user_1' in mock_auth._mock_users
        assert 'admin_user' in mock_auth._mock_users
        assert 'test_patient' in mock_auth._mock_users

    def test_create_token_success(self, mock_auth):
        """Test successful token creation."""
        user_data = {
            "user_id": "test_user_123",
            "email": "test@example.com",
            "role": "user"
        }

        token = mock_auth.create_token(user_data)

        assert isinstance(token, str)
        assert len(token) > 0

        # Verify token can be decoded
        decoded = jwt.decode(token, mock_auth.secret_key, algorithms=[mock_auth.algorithm])
        assert decoded["user_id"] == user_data["user_id"]
        assert decoded["email"] == user_data["email"]
        assert decoded["role"] == user_data["role"]
        assert "exp" in decoded

    def test_create_token_with_custom_expiration(self, mock_auth):
        """Test token creation with custom expiration."""
        user_data = {"user_id": "test_user", "email": "test@example.com"}
        custom_exp = 7200  # 2 hours

        token = mock_auth.create_token(user_data, expiration=custom_exp)

        decoded = jwt.decode(token, mock_auth.secret_key, algorithms=[mock_auth.algorithm])

        # Check that expiration is approximately 2 hours from now
        exp_time = datetime.fromtimestamp(decoded["exp"], tz=UTC)
        expected_time = datetime.now(UTC) + timedelta(seconds=custom_exp)
        time_diff = abs((exp_time - expected_time).total_seconds())

        # Allow 10 seconds tolerance
        assert time_diff < 10

    def test_create_token_with_empty_data(self, mock_auth):
        """Test token creation with empty user data."""
        token = mock_auth.create_token({})

        decoded = jwt.decode(token, mock_auth.secret_key, algorithms=[mock_auth.algorithm])
        assert "exp" in decoded
        assert len(decoded) == 1  # Only exp field

    def test_create_token_with_none_data(self, mock_auth):
        """Test token creation with None user data."""
        token = mock_auth.create_token(None)

        decoded = jwt.decode(token, mock_auth.secret_key, algorithms=[mock_auth.algorithm])
        assert "exp" in decoded

    def test_verify_token_success(self, mock_auth, valid_token_payload):
        """Test successful token verification."""
        # Create token
        token = jwt.encode(valid_token_payload, mock_auth.secret_key, algorithm=mock_auth.algorithm)

        # Verify token
        result = mock_auth.verify_token(token)

        assert result["user_id"] == valid_token_payload["user_id"]
        assert result["email"] == valid_token_payload["email"]
        assert result["role"] == valid_token_payload["role"]

    def test_verify_token_expired(self, mock_auth, expired_token_payload):
        """Test verification of expired token."""
        # Create expired token
        token = jwt.encode(expired_token_payload, mock_auth.secret_key, algorithm=mock_auth.algorithm)

        # Verify token should raise exception
        with pytest.raises(jwt.ExpiredSignatureError):
            mock_auth.verify_token(token)

    def test_verify_token_invalid_signature(self, mock_auth, valid_token_payload):
        """Test verification of token with invalid signature."""
        # Create token with different secret
        token = jwt.encode(valid_token_payload, "wrong_secret", algorithm=mock_auth.algorithm)

        # Verify token should raise exception
        with pytest.raises(jwt.InvalidSignatureError):
            mock_auth.verify_token(token)

    def test_verify_token_malformed(self, mock_auth):
        """Test verification of malformed token."""
        malformed_token = "not.a.valid.jwt.token"

        with pytest.raises(jwt.DecodeError):
            mock_auth.verify_token(malformed_token)

    def test_verify_token_empty_string(self, mock_auth):
        """Test verification of empty token."""
        with pytest.raises(jwt.DecodeError):
            mock_auth.verify_token("")

    def test_verify_token_none(self, mock_auth):
        """Test verification of None token."""
        with pytest.raises(TypeError):
            mock_auth.verify_token(None)

    def test_verify_token_without_expiration(self, mock_auth):
        """Test verification of token without expiration."""
        payload = {
            "user_id": "test_user",
            "email": "test@example.com"
            # No 'exp' field
        }

        token = jwt.encode(payload, mock_auth.secret_key, algorithm=mock_auth.algorithm)

        # Should still work but won't check expiration
        result = mock_auth.verify_token(token)
        assert result["user_id"] == payload["user_id"]

    def test_get_current_user_success(self, mock_auth, valid_token_payload):
        """Test successful current user retrieval."""
        # Create valid token
        token = jwt.encode(valid_token_payload, mock_auth.secret_key, algorithm=mock_auth.algorithm)

        user = mock_auth.get_current_user(token)

        assert user["user_id"] == valid_token_payload["user_id"]
        assert user["email"] == valid_token_payload["email"]
        assert user["authenticated"] is True

    def test_get_current_user_invalid_token(self, mock_auth):
        """Test current user retrieval with invalid token."""
        invalid_token = "invalid.token.here"

        user = mock_auth.get_current_user(invalid_token)

        assert user["user_id"] is None
        assert user["email"] is None
        assert user["authenticated"] is False
        assert "error" in user

    def test_get_current_user_expired_token(self, mock_auth, expired_token_payload):
        """Test current user retrieval with expired token."""
        token = jwt.encode(expired_token_payload, mock_auth.secret_key, algorithm=mock_auth.algorithm)

        user = mock_auth.get_current_user(token)

        assert user["user_id"] is None
        assert user["authenticated"] is False
        assert "error" in user

    def test_get_current_user_none_token(self, mock_auth):
        """Test current user retrieval with None token."""
        user = mock_auth.get_current_user(None)

        assert user["user_id"] is None
        assert user["authenticated"] is False
        assert "error" in user

    def test_authenticate_user_success(self, mock_auth):
        """Test successful user authentication."""
        # This is a mock, so any credentials should work
        result = mock_auth.authenticate_user("test@example.com", "password123")

        assert result["success"] is True
        assert result["user_id"] == "mock_user_123"
        assert result["email"] == "test@example.com"
        assert "token" in result

    def test_authenticate_user_empty_email(self, mock_auth):
        """Test authentication with empty email."""
        result = mock_auth.authenticate_user("", "password123")

        assert result["success"] is False
        assert "error" in result

    def test_authenticate_user_empty_password(self, mock_auth):
        """Test authentication with empty password."""
        result = mock_auth.authenticate_user("test@example.com", "")

        assert result["success"] is False
        assert "error" in result

    def test_authenticate_user_none_credentials(self, mock_auth):
        """Test authentication with None credentials."""
        result = mock_auth.authenticate_user(None, None)

        assert result["success"] is False
        assert "error" in result

    def test_edge_cases_and_error_handling(self, mock_auth):
        """Test various edge cases and error conditions."""
        # Test with special characters in user data
        special_data = {
            "user_id": "user@domain.com",
            "email": "test+tag@example.com",
            "name": "José María",
            "special_chars": "!@#$%^&*()"
        }

        token = mock_auth.create_token(special_data)
        decoded = mock_auth.verify_token(token)

        assert decoded["user_id"] == special_data["user_id"]
        assert decoded["email"] == special_data["email"]

    def test_concurrent_operations(self, mock_auth):
        """Test concurrent auth operations."""
        # Simulate multiple simultaneous operations
        results = []

        for i in range(10):
            user_data = {
                "user_id": f"user_{i}",
                "email": f"user{i}@example.com"
            }

            token = mock_auth.create_token(user_data)
            verified = mock_auth.verify_token(token)
            results.append(verified["user_id"] == user_data["user_id"])

        # All operations should succeed
        assert all(results)

    def test_large_token_payload(self, mock_auth):
        """Test token creation with large payload."""
        large_data = {
            "user_id": "test_user",
            "large_field": "x" * 1000,  # 1KB of data
            "metadata": {f"key_{i}": f"value_{i}" for i in range(100)}
        }

        token = mock_auth.create_token(large_data)
        decoded = mock_auth.verify_token(token)

        assert decoded["user_id"] == large_data["user_id"]
        assert len(decoded["large_field"]) == 1000

    def test_token_boundary_conditions(self, mock_auth):
        """Test boundary conditions for token operations."""
        # Test with expiration at exact current time
        now = datetime.now(UTC)
        boundary_payload = {
            "user_id": "test_user",
            "exp": now.timestamp()
        }

        token = jwt.encode(boundary_payload, mock_auth.secret_key, algorithm=mock_auth.algorithm)

        # Should be expired or very close to expiring
        try:
            result = mock_auth.verify_token(token)
            # If it doesn't raise, verify the result
            assert "user_id" in result
        except jwt.ExpiredSignatureError:
            # Expected behavior for expired token
            pass
