"""CLARITY Digital Twin Platform - Startup Diagnostic Tool.

🔍 STARTUP HANG DIAGNOSTICS
This script tests each startup component individually with timeouts to identify
which component is causing the application to hang during lifespan initialization.

Usage:
    python debug_startup.py

The script will test each component and report which one hangs.
"""

import asyncio
from pathlib import Path
import sys
import time

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


async def debug_startup() -> str | None:
    """Main diagnostic function to identify startup hang points."""
    # Test each startup component with timeouts
    components = [
        ("Logging setup", setup_logging_test),
        ("Config provider", test_config_provider),
        ("Auth provider init", test_auth_provider_init),
        ("Repository init", test_repository_init),
        ("Full lifespan simulation", test_full_lifespan),
    ]

    overall_start = time.perf_counter()
    culprit = None

    for name, test_func in components:
        start = time.perf_counter()

        try:
            await asyncio.wait_for(test_func(), timeout=10.0)
            elapsed = time.perf_counter() - start

        except TimeoutError:
            elapsed = time.perf_counter() - start
            culprit = name
            break

        except Exception as e:
            elapsed = time.perf_counter() - start
            culprit = name
            break

    total_elapsed = time.perf_counter() - overall_start

    return culprit


async def setup_logging_test() -> None:
    """Test logging setup component."""
    from clarity.core.logging_config import setup_logging

    setup_logging()


async def test_config_provider() -> None:
    """Test config provider initialization."""
    from clarity.core.container import get_container

    container = get_container()
    config = container.get_config_provider()

    # Test some config access
    is_dev = config.is_development()
    auth_enabled = config.is_auth_enabled()
    firebase_config = config.get_firebase_config()


async def test_auth_provider_init() -> None:
    """Test auth provider initialization."""
    from clarity.core.container import get_container

    container = get_container()

    auth = container.get_auth_provider()

    if hasattr(auth, "initialize"):
        await auth.initialize()


async def test_repository_init() -> None:
    """Test repository initialization."""
    from clarity.core.container import get_container

    container = get_container()

    repo = container.get_health_data_repository()

    if hasattr(repo, "initialize"):
        await repo.initialize()


async def test_full_lifespan() -> None:
    """Test the full lifespan context manager."""
    from clarity.core.container import get_container

    container = get_container()

    # Simulate what the lifespan does
    from clarity.core.logging_config import setup_logging

    setup_logging()

    # Initialize auth provider
    auth_provider = container.get_auth_provider()
    if hasattr(auth_provider, "initialize"):
        await auth_provider.initialize()

    # Initialize repository
    repository = container.get_health_data_repository()
    if hasattr(repository, "initialize"):
        await repository.initialize()


async def test_environment_variables() -> None:
    """Test environment variable availability."""
    import os

    env_vars_to_check = [
        "FIREBASE_PROJECT_ID",
        "FIREBASE_CREDENTIALS_PATH",
        "GCP_PROJECT_ID",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "ENVIRONMENT",
        "DEBUG",
        "ENABLE_AUTH",
    ]

    for var in env_vars_to_check:
        value = os.getenv(var)
        if value:
            pass


async def test_file_permissions() -> None:
    """Test file access permissions."""
    import os
    from pathlib import Path

    # Check common credential file locations
    credential_paths = [
        "credentials/firebase-credentials.json",
        "credentials/service-account.json",
        ".env",
        "src/clarity",
    ]

    for path_str in credential_paths:
        path = Path(path_str)
        if path.exists() and path.is_file():
            readable = os.access(path, os.R_OK)


if __name__ == "__main__":

    try:
        # First check environment and files
        asyncio.run(test_environment_variables())
        asyncio.run(test_file_permissions())

        # Main diagnostic
        culprit = asyncio.run(debug_startup())

        if culprit:

            if "Auth provider" in culprit or "Repository" in culprit or "Config provider" in culprit or "Logging" in culprit:
                pass

    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback

        traceback.print_exc()
