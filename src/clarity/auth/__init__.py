"""CLARITY Digital Twin Platform - Authentication Package.

Firebase-based authentication and authorization for enterprise healthcare applications.
"""

from clarity.auth.decorators import require_auth
from clarity.auth.dependencies import (
    get_current_user_from_context,
    get_current_user_from_context_required,
)
from clarity.auth.firebase_middleware import FirebaseAuthMiddleware, FirebaseAuthProvider
from clarity.auth.mock_auth import MockAuthProvider
from clarity.auth.modal_auth_fix import get_user_context, set_user_context

__all__ = [
    "require_auth",
    "get_current_user_from_context",
    "get_current_user_from_context_required",
    "FirebaseAuthMiddleware",
    "FirebaseAuthProvider",
    "MockAuthProvider",
    "get_user_context",
    "set_user_context",
]
