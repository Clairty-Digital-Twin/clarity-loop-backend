"""Container compatibility module - aliases to container_aws."""

from __future__ import annotations


# Import everything from container_aws for compatibility
from clarity.core.container_aws import *  # noqa: F403
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI


# Add any missing functions for tests
def create_application() -> FastAPI:
    """Create application (for tests)."""
    # Import here to avoid circular imports
    from clarity.main import app as clarity_app

    return clarity_app
