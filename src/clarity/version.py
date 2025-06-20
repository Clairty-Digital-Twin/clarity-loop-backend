"""Version information for clarity backend."""

from importlib import metadata


def get_version() -> str:
    """Get the current version of the application.

    Returns:
        str: Version string from package metadata, or fallback value
    """
    try:
        return metadata.version("clarity-loop-backend")
    except metadata.PackageNotFoundError:
        # Fallback for development environments where package isn't installed
        return "0.1.0-dev"
