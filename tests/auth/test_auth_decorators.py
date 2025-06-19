"""Tests for authentication decorators."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import HTTPException, status
import pytest

from clarity.auth.decorators import require_auth, require_permission, require_role
from clarity.models.user import User


@pytest.mark.asyncio
async def test_require_auth_no_user() -> None:
    """Test that require_auth raises 401 if no user is present."""

    @require_auth()
    async def dummy_endpoint(  # noqa: RUF029
        current_user: User | None = None,
    ) -> str:
        _ = current_user  # Used by decorator
        return "OK"

    with pytest.raises(HTTPException) as excinfo:
        await dummy_endpoint(current_user=None)

    assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert excinfo.value.detail == "Authentication required"


@pytest.mark.asyncio
async def test_require_auth_with_user() -> None:
    """Test that require_auth succeeds if a user is present."""
    mock_user = MagicMock(spec=User)

    @require_auth()
    async def dummy_endpoint(  # noqa: RUF029
        current_user: User | None = None,
    ) -> str:
        _ = current_user  # Used by decorator
        return "OK"

    result = await dummy_endpoint(current_user=mock_user)
    assert result == "OK"


@pytest.mark.asyncio
async def test_require_permission_no_user() -> None:
    """Test that require_permission raises 401 if no user is present."""

    @require_permission("some_permission")
    async def dummy_endpoint(  # noqa: RUF029
        current_user: User | None = None,  # noqa: ARG001
    ) -> str:
        return "OK"

    with pytest.raises(HTTPException) as excinfo:
        await dummy_endpoint(current_user=None)

    assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert excinfo.value.detail == "Authentication required"


@pytest.mark.asyncio
async def test_require_permission_with_user() -> None:
    """Test that require_permission succeeds if a user is present (no-op check)."""
    mock_user = MagicMock(spec=User)

    @require_permission("some_permission")
    async def dummy_endpoint(  # noqa: RUF029
        current_user: User | None = None,  # noqa: ARG001
    ) -> str:
        return "OK"

    result = await dummy_endpoint(current_user=mock_user)
    assert result == "OK"


@pytest.mark.asyncio
async def test_require_role_no_user() -> None:
    """Test that require_role raises 401 if no user is present."""

    @require_role("some_role")
    async def dummy_endpoint(  # noqa: RUF029
        current_user: User | None = None,  # noqa: ARG001
    ) -> str:
        return "OK"

    with pytest.raises(HTTPException) as excinfo:
        await dummy_endpoint(current_user=None)

    assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert excinfo.value.detail == "Authentication required"


@pytest.mark.asyncio
async def test_require_role_with_user() -> None:
    """Test that require_role succeeds if a user is present (no-op check)."""
    mock_user = MagicMock(spec=User)

    @require_role("some_role")
    async def dummy_endpoint(  # noqa: RUF029
        current_user: User | None = None,  # noqa: ARG001
    ) -> str:
        return "OK"

    result = await dummy_endpoint(current_user=mock_user)
    assert result == "OK"
