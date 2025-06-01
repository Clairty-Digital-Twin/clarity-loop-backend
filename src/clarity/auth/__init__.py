"""CLARITY Authentication Module

Firebase-based authentication and authorization system for HIPAA-compliant
health data access control.
"""

from .firebase_auth import (
    FirebaseAuthMiddleware,
    get_current_user,
    require_auth,
    require_permission,
    require_role,
)
from .models import (
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
