"""Comprehensive tests for MockAuthProvider.

Tests all methods and edge cases to improve coverage from 37% to 90%+.
"""

# Comprehensive testing imports

import pytest

from clarity.auth.mock_auth import MockAuthProvider


class TestMockAuthProviderComprehensive:
    """Comprehensive test coverage for MockAuthProvider."""

    @pytest.fixture
    def mock_auth_provider(self):
        """Create MockAuthProvider instance for testing."""
        return MockAuthProvider()

    def test_initialization(self, mock_auth_provider):
        """Test MockAuthProvider initialization."""
        assert hasattr(mock_auth_provider, '_mock_users')
        assert len(mock_auth_provider._mock_users) == 3
        assert 'mock_user_1' in mock_auth_provider._mock_users
        assert 'admin_user' in mock_auth_provider._mock_users
        assert 'test_patient' in mock_auth_provider._mock_users

    def test_initialization_user_data_structure(self, mock_auth_provider):
        """Test that mock users have correct structure."""
        for user_id, user_data in mock_auth_provider._mock_users.items():
            assert 'user_id' in user_data
            assert 'email' in user_data
            assert 'name' in user_data
            assert 'roles' in user_data
            assert 'verified' in user_data
            assert user_data['user_id'] == user_id
            assert isinstance(user_data['roles'], list)
            assert isinstance(user_data['verified'], bool)

    async def test_verify_token_valid_dev_token_user(self, mock_auth_provider):
        """Test token verification with valid dev_token_user."""
        result = await mock_auth_provider.verify_token("dev_token_user")

        assert result is not None
        assert result['user_id'] == 'mock_user_1'
        assert result['email'] == 'developer@clarity.health'
        assert result['name'] == 'Development User'
        assert 'user' in result['roles']
        assert 'developer' in result['roles']
        assert result['verified'] is True

    async def test_verify_token_valid_dev_token_admin(self, mock_auth_provider):
        """Test token verification with valid dev_token_admin."""
        result = await mock_auth_provider.verify_token("dev_token_admin")

        assert result is not None
        assert result['user_id'] == 'admin_user'
        assert result['email'] == 'admin@clarity.health'
        assert result['name'] == 'Admin User'
        assert 'admin' in result['roles']
        assert 'user' in result['roles']
        assert result['verified'] is True

    async def test_verify_token_valid_dev_token_patient(self, mock_auth_provider):
        """Test token verification with valid dev_token_patient."""
        result = await mock_auth_provider.verify_token("dev_token_patient")

        assert result is not None
        assert result['user_id'] == 'test_patient'
        assert result['email'] == 'patient@clarity.health'
        assert result['name'] == 'Test Patient'
        assert 'patient' in result['roles']
        assert result['verified'] is True

    async def test_verify_token_valid_mock_token(self, mock_auth_provider):
        """Test token verification with valid mock_token."""
        result = await mock_auth_provider.verify_token("mock_token")

        assert result is not None
        assert result['user_id'] == 'mock_user_1'
        assert result['email'] == 'developer@clarity.health'

    async def test_verify_token_invalid_token(self, mock_auth_provider):
        """Test token verification with invalid token."""
        result = await mock_auth_provider.verify_token("invalid_token")

        assert result is None

    async def test_verify_token_empty_string(self, mock_auth_provider):
        """Test token verification with empty string."""
        result = await mock_auth_provider.verify_token("")

        assert result is None

    async def test_verify_token_none(self, mock_auth_provider):
        """Test token verification with None token."""
        result = await mock_auth_provider.verify_token(None)

        assert result is None

    async def test_verify_token_whitespace(self, mock_auth_provider):
        """Test token verification with whitespace token."""
        result = await mock_auth_provider.verify_token("   ")

        assert result is None

    async def test_verify_token_case_sensitive(self, mock_auth_provider):
        """Test that token verification is case sensitive."""
        result = await mock_auth_provider.verify_token("DEV_TOKEN_USER")

        assert result is None

    async def test_verify_token_returns_copy(self, mock_auth_provider):
        """Test that verify_token returns a copy of user data."""
        result1 = await mock_auth_provider.verify_token("dev_token_user")
        result2 = await mock_auth_provider.verify_token("dev_token_user")

        # Should be equal but not the same object
        assert result1 == result2
        assert result1 is not result2

        # Modifying one shouldn't affect the other
        result1['modified'] = True
        assert 'modified' not in result2

    async def test_get_user_info_valid_user_id(self, mock_auth_provider):
        """Test get_user_info with valid user ID."""
        result = await mock_auth_provider.get_user_info("mock_user_1")

        assert result is not None
        assert result['user_id'] == 'mock_user_1'
        assert result['email'] == 'developer@clarity.health'
        assert result['name'] == 'Development User'

    async def test_get_user_info_admin_user(self, mock_auth_provider):
        """Test get_user_info with admin user ID."""
        result = await mock_auth_provider.get_user_info("admin_user")

        assert result is not None
        assert result['user_id'] == 'admin_user'
        assert result['email'] == 'admin@clarity.health'
        assert 'admin' in result['roles']

    async def test_get_user_info_patient_user(self, mock_auth_provider):
        """Test get_user_info with patient user ID."""
        result = await mock_auth_provider.get_user_info("test_patient")

        assert result is not None
        assert result['user_id'] == 'test_patient'
        assert result['email'] == 'patient@clarity.health'
        assert 'patient' in result['roles']

    async def test_get_user_info_invalid_user_id(self, mock_auth_provider):
        """Test get_user_info with invalid user ID."""
        result = await mock_auth_provider.get_user_info("nonexistent_user")

        assert result is None

    async def test_get_user_info_empty_string(self, mock_auth_provider):
        """Test get_user_info with empty string."""
        result = await mock_auth_provider.get_user_info("")

        assert result is None

    async def test_get_user_info_none(self, mock_auth_provider):
        """Test get_user_info with None."""
        result = await mock_auth_provider.get_user_info(None)

        assert result is None

    async def test_get_user_info_returns_copy(self, mock_auth_provider):
        """Test that get_user_info returns a copy of user data."""
        result1 = await mock_auth_provider.get_user_info("mock_user_1")
        result2 = await mock_auth_provider.get_user_info("mock_user_1")

        # Should be equal but not the same object
        assert result1 == result2
        assert result1 is not result2

        # Modifying one shouldn't affect the other
        result1['modified'] = True
        assert 'modified' not in result2

    async def test_initialize(self, mock_auth_provider):
        """Test initialize method."""
        # Should complete without error
        await mock_auth_provider.initialize()
        # State should remain unchanged
        assert len(mock_auth_provider._mock_users) == 3

    async def test_cleanup(self, mock_auth_provider):
        """Test cleanup method."""
        # Should complete without error
        await mock_auth_provider.cleanup()
        # State should remain unchanged for mock
        assert len(mock_auth_provider._mock_users) == 3

    def test_create_mock_token_admin_user(self, mock_auth_provider):
        """Test create_mock_token for admin user."""
        token = mock_auth_provider.create_mock_token("admin_user")

        assert token == "dev_token_admin"

    def test_create_mock_token_test_patient(self, mock_auth_provider):
        """Test create_mock_token for test patient."""
        token = mock_auth_provider.create_mock_token("test_patient")

        assert token == "dev_token_patient"

    def test_create_mock_token_default_user(self, mock_auth_provider):
        """Test create_mock_token for default user."""
        token = mock_auth_provider.create_mock_token("mock_user_1")

        assert token == "dev_token_user"

    def test_create_mock_token_unknown_user(self, mock_auth_provider):
        """Test create_mock_token for unknown user."""
        token = mock_auth_provider.create_mock_token("unknown_user")

        assert token == "dev_token_user"

    def test_create_mock_token_empty_string(self, mock_auth_provider):
        """Test create_mock_token with empty string."""
        token = mock_auth_provider.create_mock_token("")

        assert token == "dev_token_user"

    def test_create_mock_token_none(self, mock_auth_provider):
        """Test create_mock_token with None."""
        token = mock_auth_provider.create_mock_token(None)

        assert token == "dev_token_user"

    def test_get_available_mock_users(self, mock_auth_provider):
        """Test get_available_mock_users method."""
        users = mock_auth_provider.get_available_mock_users()

        assert isinstance(users, list)
        assert len(users) == 3

        # Check that all expected users are present
        user_ids = [user['user_id'] for user in users]
        assert 'mock_user_1' in user_ids
        assert 'admin_user' in user_ids
        assert 'test_patient' in user_ids

    def test_get_available_mock_users_structure(self, mock_auth_provider):
        """Test structure of users returned by get_available_mock_users."""
        users = mock_auth_provider.get_available_mock_users()

        for user in users:
            assert 'user_id' in user
            assert 'email' in user
            assert 'name' in user
            assert 'roles' in user
            assert 'verified' in user
            assert isinstance(user['roles'], list)
            assert isinstance(user['verified'], bool)

    def test_get_available_mock_users_returns_copy(self, mock_auth_provider):
        """Test that get_available_mock_users returns copies."""
        users1 = mock_auth_provider.get_available_mock_users()
        users2 = mock_auth_provider.get_available_mock_users()

        # Should be equal but not the same objects
        assert users1 == users2
        assert users1 is not users2

        # Modifying one shouldn't affect the other
        if len(users1) > 0:
            original_email = users1[0]['email']
            users1[0]['email'] = 'modified@test.com'
            # Verify the second call returns unmodified data
            assert users2[0]['email'] == original_email

    async def test_integration_verify_then_get_user_info(self, mock_auth_provider):
        """Test integration between verify_token and get_user_info."""
        # Verify token to get user info
        token_result = await mock_auth_provider.verify_token("dev_token_admin")
        assert token_result is not None

        user_id = token_result['user_id']

        # Use user_id to get user info directly
        user_info_result = await mock_auth_provider.get_user_info(user_id)
        assert user_info_result is not None

        # Results should match
        assert token_result['user_id'] == user_info_result['user_id']
        assert token_result['email'] == user_info_result['email']
        assert token_result['name'] == user_info_result['name']
        assert token_result['roles'] == user_info_result['roles']

    async def test_integration_create_token_then_verify(self, mock_auth_provider):
        """Test integration between create_mock_token and verify_token."""
        # Create token for admin user
        token = mock_auth_provider.create_mock_token("admin_user")

        # Verify the created token
        result = await mock_auth_provider.verify_token(token)

        assert result is not None
        assert result['user_id'] == 'admin_user'

    async def test_mock_users_immutability(self, mock_auth_provider):
        """Test that internal mock users data is protected."""
        # Get user info
        user_info = await mock_auth_provider.get_user_info("mock_user_1")

        # Modify the returned data
        original_email = user_info['email']
        user_info['email'] = 'modified@example.com'

        # Get user info again - should be unchanged
        user_info_again = await mock_auth_provider.get_user_info("mock_user_1")
        assert user_info_again['email'] == original_email

    async def test_all_token_types_map_to_valid_users(self, mock_auth_provider):
        """Test that all predefined tokens map to valid users."""
        token_mappings = {
            "dev_token_user": "mock_user_1",
            "dev_token_admin": "admin_user",
            "dev_token_patient": "test_patient",
            "mock_token": "mock_user_1"
        }

        for token, expected_user_id in token_mappings.items():
            result = await mock_auth_provider.verify_token(token)
            assert result is not None
            assert result['user_id'] == expected_user_id

    async def test_user_roles_are_correctly_assigned(self, mock_auth_provider):
        """Test that each user has the correct roles assigned."""
        expected_roles = {
            'mock_user_1': ['user', 'developer'],
            'admin_user': ['admin', 'user'],
            'test_patient': ['patient']
        }

        for user_id, expected in expected_roles.items():
            user_info = await mock_auth_provider.get_user_info(user_id)
            assert user_info is not None
            assert set(user_info['roles']) == set(expected)

    async def test_all_users_are_verified(self, mock_auth_provider):
        """Test that all mock users have verified status."""
        for user_id in mock_auth_provider._mock_users:
            user_info = await mock_auth_provider.get_user_info(user_id)
            assert user_info is not None
            assert user_info['verified'] is True
