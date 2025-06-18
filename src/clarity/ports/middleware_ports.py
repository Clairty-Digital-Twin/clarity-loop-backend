"""Middleware port interfaces.

Defines the contract for middleware components following Clean Architecture.
Business logic layer depends on this abstraction, not concrete implementations.
"""

# removed – breaks FastAPI

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from fastapi import Request, Response


class IMiddleware(ABC):
    """Interface for HTTP middleware components.

    Following Clean Architecture:
    - Middleware operates at the interface adapter layer
    - Handles cross-cutting concerns (auth, logging, etc.)
    - Should not contain business logic
    """

    @abstractmethod
    async def __call__(
        self, request: Request, call_next: Awaitable[Response]
    ) -> Response:
        """Process request through middleware.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware or route handler in the chain

        Returns:
            The HTTP response after processing
        """
