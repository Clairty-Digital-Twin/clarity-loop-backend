"""CLARITY Digital Twin Platform - Authentication Package.

Firebase-based authentication and authorization for enterprise healthcare applications.
"""

from clarity.auth.decorators import require_auth
from clarity.auth.dependencies import (
    AuthenticatedUser,
    OptionalUser,
    get_authenticated_user,
    get_optional_user,
)
from clarity.auth.firebase_middleware import (
    FirebaseAuthMiddleware,
    FirebaseAuthProvider,
)
from clarity.auth.mock_auth import MockAuthProvider
from clarity.auth.modal_auth_fix import get_user_context, set_user_context
from clarity.models.auth import UserContext

__all__ = [
    "AuthenticatedUser",
    "FirebaseAuthMiddleware",
    "FirebaseAuthProvider",
    "MockAuthProvider",
    "OptionalUser",
    "UserContext",
    "get_authenticated_user",
    "get_optional_user",
    "get_user_context",
    "require_auth",
    "set_user_context",
]
