"""Authentication decorators and utilities for API endpoints."""

from functools import wraps
from typing import Callable, Any

from fastapi import Depends, HTTPException, status

from clarity.auth.firebase_auth import get_current_user_required
from clarity.models.user import User


def require_auth(func: Callable) -> Callable:
    """Decorator to require authentication for an endpoint."""
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # The dependency injection will handle authentication
        return await func(*args, **kwargs)
    
    return wrapper


def require_permission(permission: str):
    """Decorator to require specific permission for an endpoint."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user from kwargs (injected by dependency)
            user = kwargs.get('current_user')
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # Check permission (simplified - in real app check user roles/permissions)
            # For now, all authenticated users have all permissions
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_role(role: str):
    """Decorator to require specific role for an endpoint."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user from kwargs (injected by dependency)
            user = kwargs.get('current_user')
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # Check role (simplified - in real app check user roles)
            # For now, all authenticated users have all roles
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator