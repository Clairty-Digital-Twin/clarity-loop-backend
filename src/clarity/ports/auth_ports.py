"""Authentication port interfaces.

Defines the contract for authentication providers following Clean Architecture.
Business logic layer depends on this abstraction, not concrete implementations.
"""

# removed - breaks FastAPI

from abc import ABC, abstractmethod
from typing import Any


class IAuthProvider(ABC):
    """Abstract authentication provider interface.

    Following Uncle Bob's Clean Architecture principles, this interface
    allows the business logic layer to depend on abstractions rather than
    concrete authentication implementations.
    """

    @abstractmethod
    async def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify an authentication token.

        Args:
            token: The authentication token to verify

        Returns:
            User information dictionary if valid, None if invalid
        """

    @abstractmethod
    async def get_user_info(self, user_id: str) -> dict[str, Any] | None:
        """Get user information by ID.

        Args:
            user_id: The user identifier

        Returns:
            User information dictionary if found, None if not found
        """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the authentication provider.

        Performs any necessary setup operations like loading configuration,
        establishing connections, etc.
        """

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up authentication provider resources.

        Performs cleanup operations like closing connections, releasing resources, etc.
        """

    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the provider is initialized.

        Returns:
            True if initialized, False otherwise
        """
