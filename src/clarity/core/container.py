"""Container compatibility module - aliases to container_aws."""

# removed - breaks FastAPI

from typing import TYPE_CHECKING

# Import everything from container_aws for compatibility
from clarity.core.container_aws import *  # noqa: F403

if TYPE_CHECKING:
    from fastapi import FastAPI


# Add any missing functions for tests
def create_application() -> FastAPI:
    """Create application (for tests)."""
    # Import here to avoid circular imports
    from clarity.main import app as clarity_app  # noqa: PLC0415

    return clarity_app
