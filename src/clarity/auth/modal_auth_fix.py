"""Modal-specific authentication fix using contextvars.

Modal doesn't properly propagate request.state between middleware and route handlers.
This module provides a workaround using Python's contextvars for thread-local storage.
"""

# removed - breaks FastAPI

import contextvars
from typing import TYPE_CHECKING

from clarity.models.auth import UserContext

if TYPE_CHECKING:
    pass  # Only for type stubs now

# Thread-local storage for user context
_user_context: contextvars.ContextVar[UserContext | None] = contextvars.ContextVar(
    "user_context", default=None
)


def set_user_context(user: UserContext) -> None:
    """Store user context in thread-local storage."""
    _user_context.set(user)


def get_user_context() -> UserContext | None:
    """Retrieve user context from thread-local storage."""
    return _user_context.get()


def clear_user_context() -> None:
    """Clear user context from thread-local storage."""
    _user_context.set(None)
