"""
CLARITY Digital Twin Platform - Authentication Module

Firebase-based authentication and authorization for secure health data access.
"""

from .firebase_auth import FirebaseAuthMiddleware, get_current_user, require_auth
from .models import UserContext, AuthError

__all__ = [
    "FirebaseAuthMiddleware",
    "get_current_user", 
    "require_auth",
    "UserContext",
    "AuthError"
]
