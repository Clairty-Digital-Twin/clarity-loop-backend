"""Authentication decorators and utilities for API endpoints."""

from collections.abc import Callable
from functools import wraps
from typing import Any

from fastapi import HTTPException, status


def require_auth(
    permissions: list[str] | None = None, roles: list[str] | None = None
) -> Callable[..., Any]:
    """Decorator to require authentication and optionally check permissions/roles."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract user from kwargs (injected by dependency)
            user = kwargs.get("current_user")
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            # Check permissions if specified
            if permissions:
                # For now, all authenticated users have all permissions
                # In a real app, check user.permissions or roles
                pass

            # Check roles if specified
            if roles:
                # For now, all authenticated users have all roles
                # In a real app, check user.roles
                pass

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_permission(permission: str) -> Callable[..., Any]:
    """Decorator to require specific permission for an endpoint."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract user from kwargs (injected by dependency)
            user = kwargs.get("current_user")
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            # Check permission (simplified - in real app check user roles/permissions)
            # For now, all authenticated users have all permissions
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_role(role: str) -> Callable[..., Any]:
    """Decorator to require specific role for an endpoint."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract user from kwargs (injected by dependency)
            user = kwargs.get("current_user")
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            # Check role (simplified - in real app check user roles)
            # For now, all authenticated users have all roles
            return await func(*args, **kwargs)

        return wrapper

    return decorator
