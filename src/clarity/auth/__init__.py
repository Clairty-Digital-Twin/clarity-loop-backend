"""CLARITY Digital Twin Platform - Authentication Package.

Firebase-based authentication and authorization for enterprise healthcare applications.
"""

from clarity.auth.decorators import (
    require_auth,
    require_permission,
    require_role,
)
from clarity.auth.dependencies import (
    AuthenticatedUser,
    OptionalUser,
    get_authenticated_user,
    get_optional_user,
    require_active_account,
    require_verified_email,
)
from clarity.auth.firebase_auth import (
    get_current_user,
    get_current_user_context,
    get_current_user_context_required,
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
    # New clean dependencies
    "AuthenticatedUser",
    "OptionalUser",
    "get_authenticated_user",
    "get_optional_user",
    "require_active_account",
    "require_verified_email",
    # Legacy dependencies (for backward compatibility)
    "get_current_user",
    "get_current_user_context",
    "get_current_user_context_required",
    "get_current_user_required",
    "get_current_user_websocket",
    # Decorators
    "require_auth",
    "require_permission",
    "require_role",
    # Core classes
    "AuthError",
    "FirebaseAuthMiddleware",
    "FirebaseAuthProvider",
    "Permission",
    "TokenInfo",
    "UserContext",
    "UserRole",
]
