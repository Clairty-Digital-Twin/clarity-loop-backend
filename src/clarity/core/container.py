"""Container compatibility module - aliases to container_aws."""

# Import everything from container_aws for compatibility
from clarity.core.container_aws import *  # noqa: F403


# Add any missing functions for tests
def create_application():
    """Create application (for tests)."""
    from clarity.main import app

    return app
