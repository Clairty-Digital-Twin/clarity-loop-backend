"""Mock Authentication Provider for Development.

Following Clean Architecture and SOLID principles, this module provides
a mock implementation of IAuthProvider interface for development and testing.
Implements Liskov Substitution Principle - can substitute real auth provider.
"""

# removed - breaks FastAPI

from typing import Any

from clarity.ports.auth_ports import IAuthProvider


class MockAuthProvider(IAuthProvider):
    """Mock authentication provider for development and testing.

    Follows Single Responsibility Principle - only handles mock authentication.
    Implements Open/Closed Principle - extends IAuthProvider without modification.
    """

    def __init__(self) -> None:
        """Initialize mock authentication provider."""
        # Predefined mock users for development
        self._mock_users = {
            "mock_user_1": {
                "user_id": "mock_user_1",
                "email": "developer@clarity.health",
                "name": "Development User",
                "roles": ["user", "developer"],
                "verified": True,
            },
            "admin_user": {
                "user_id": "admin_user",
                "email": "admin@clarity.health",
                "name": "Admin User",
                "roles": ["admin", "user"],
                "verified": True,
            },
            "test_patient": {
                "user_id": "test_patient",
                "email": "patient@clarity.health",
                "name": "Test Patient",
                "roles": ["patient"],
                "verified": True,
            },
        }

    async def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify authentication token (mock implementation).

        Args:
            token: Authentication token to verify

        Returns:
            User information if token is valid, None otherwise
        """
        # Mock token validation - accepts specific development tokens
        mock_tokens = {
            "dev_token_user": "mock_user_1",
            "dev_token_admin": "admin_user",
            "dev_token_patient": "test_patient",
            "mock_token": "mock_user_1",  # Default development token
        }

        user_id = mock_tokens.get(token)
        if user_id and user_id in self._mock_users:
            return self._mock_users[user_id].copy()

        return None

    async def get_user_info(self, user_id: str) -> dict[str, Any] | None:
        """Get user information by ID (mock implementation).

        Args:
            user_id: User identifier

        Returns:
            User information if found, None otherwise
        """
        return (
            self._mock_users.get(user_id, {}).copy()
            if user_id in self._mock_users
            else None
        )

    async def initialize(self) -> None:
        """Initialize mock authentication provider.

        This method can be called during application startup.
        For mock provider, no initialization is needed.
        """

    async def cleanup(self) -> None:
        """Clean up mock authentication provider resources.

        This method can be called during application shutdown.
        For mock provider, no cleanup is needed.
        """

    @staticmethod
    def create_mock_token(user_id: str) -> str:
        """Create a mock token for testing purposes.

        Args:
            user_id: User ID to include in token

        Returns:
            Mock JWT token string
        """
        # Simple mock token creation for development
        if user_id == "admin_user":
            return "dev_token_admin"
        if user_id == "test_patient":
            return "dev_token_patient"
        return "dev_token_user"

    def get_available_mock_users(self) -> list[dict[str, Any]]:
        """Get list of available mock users for development.

        Returns:
            List of available mock users
        """
        return list(self._mock_users.values())

    def is_initialized(self) -> bool:
        """Check if the provider is initialized.

        Returns:
            True - mock provider is always ready
        """
        return True
