"""Authentication decorators and utilities for API endpoints."""

# removed - breaks FastAPI

from collections.abc import Awaitable, Callable
from functools import wraps
from typing import TYPE_CHECKING, ParamSpec, TypeVar, cast

from fastapi import HTTPException, status

from clarity.models.user import User

if TYPE_CHECKING:
    pass  # Only for type stubs now

P = ParamSpec("P")
T = TypeVar("T")


def require_auth(
    permissions: list[str] | None = None, roles: list[str] | None = None
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator to require authentication and optionally check permissions/roles."""

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Extract user from kwargs (injected by dependency)
            user = cast(User | None, kwargs.get("current_user"))
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


def require_permission(
    _permission: str,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator to require specific permission for an endpoint."""

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Extract user from kwargs (injected by dependency)
            user = cast(User | None, kwargs.get("current_user"))
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


def require_role(
    _role: str,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator to require specific role for an endpoint."""

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Extract user from kwargs (injected by dependency)
            user = cast(User | None, kwargs.get("current_user"))
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
