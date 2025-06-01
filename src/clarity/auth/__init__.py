"""CLARITY Digital Twin Platform - Authentication Package.

Firebase-based authentication and authorization for enterprise healthcare applications.
"""

from clarity.auth.firebase_auth import (
    FirebaseAuthMiddleware,
    get_current_user,
    require_auth,
    require_permission,
    require_role,
)
from clarity.auth.models import (
    AuthError,
    Permission,
    TokenInfo,
    UserContext,
    UserRole,
)

__all__ = [
    "AuthError",
    "FirebaseAuthMiddleware",
    "Permission",
    "TokenInfo",
    "UserContext",
    "UserRole",
    "get_current_user",
    "require_auth",
    "require_permission",
    "require_role",
]
