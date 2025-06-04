"""CLARITY Digital Twin Platform - Authentication Package.

Firebase-based authentication and authorization for enterprise healthcare applications.
"""

from clarity.auth.decorators import (
    require_auth,
    require_permission,
    require_role,
)
from clarity.auth.firebase_auth import (
    get_current_user,
    get_current_user_required,
    get_current_user_websocket,
)
from clarity.auth.firebase_middleware import (
    FirebaseAuthMiddleware,
    FirebaseAuthProvider,
)
from clarity.models.auth import (
    AuthError,
    Permission,
    TokenInfo,
    UserContext,
    UserRole,
)

__all__ = [
    "AuthError",
    "FirebaseAuthMiddleware",
    "FirebaseAuthProvider",
    "Permission",
    "TokenInfo",
    "UserContext",
    "UserRole",
    "get_current_user",
    "get_current_user_required",
    "get_current_user_websocket",
    "require_auth",
    "require_permission",
    "require_role",
]
