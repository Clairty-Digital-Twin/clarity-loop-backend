"""Debug script to identify startup performance bottlenecks.

This helps diagnose slow startup issues by timing each component.
"""

import asyncio
import os
from pathlib import Path
import time


def measure_import_time(module_name: str) -> float:
    """Measure time to import a module."""
    start = time.perf_counter()
    try:
        __import__(module_name)
        return time.perf_counter() - start
    except ImportError:
        return -1.0  # Indicate import failed


def find_slowest_import() -> str | None:
    """Find the slowest import to identify bottlenecks."""
    modules_to_test = [
        "clarity.core.config",
        "clarity.core.container",
        "clarity.api.v1.health_data",
        "clarity.auth.firebase_auth",
        "clarity.services.health_data_service",
        "firebase_admin",
        "google.cloud.firestore",
        "fastapi",
        "pydantic",
        "uvicorn",
    ]

    overall_start = time.perf_counter()
    slowest_time = 0.0
    culprit = None

    for name in modules_to_test:
        elapsed = measure_import_time(name)

        if elapsed > slowest_time and elapsed > 0:
            slowest_time = elapsed
            culprit = name

        if elapsed > 1.0:  # Anything over 1 second is concerning
            print(f"⚠️ Slow import detected: {name} took {elapsed:.2f}s")
            break

    total_elapsed = time.perf_counter() - overall_start
    if total_elapsed > 2.0:
        print(f"⚠️ Total import time concerning: {total_elapsed:.2f}s")

    return culprit


def test_config_access() -> None:
    """Test configuration access performance."""
    start = time.perf_counter()

    try:
        from clarity.core.config import get_settings  # noqa: PLC0415

        config = get_settings()

        # Test some config access - use available methods
        _ = config.debug
        _ = config.environment
        _ = config.firebase_project_id

        elapsed = time.perf_counter() - start
        if elapsed > 0.5:
            print(f"⚠️ Config access slow: {elapsed:.2f}s")

    except Exception as config_error:
        print(f"⚠️ Config access failed: {config_error}")


def test_container_creation() -> None:
    """Test dependency container creation performance."""
    start = time.perf_counter()

    try:
        from clarity.core.container import get_container  # noqa: PLC0415

        container = get_container()

        # Test getting some basic dependencies
        _ = container.get_config_provider()

        elapsed = time.perf_counter() - start
        if elapsed > 1.0:
            print(f"⚠️ Container creation slow: {elapsed:.2f}s")

        # Test auth provider creation
        auth_start = time.perf_counter()
        container.get_auth_provider()
        auth_elapsed = time.perf_counter() - auth_start

        if auth_elapsed > 2.0:
            print(f"⚠️ Auth provider creation slow: {auth_elapsed:.2f}s")

    except Exception as container_error:
        print(f"⚠️ Container creation failed: {container_error}")


async def test_app_creation() -> None:
    """Test FastAPI app creation performance."""
    start = time.perf_counter()

    try:
        from clarity.core.container import create_application  # noqa: PLC0415

        _ = create_application()  # Create app but don't need to use it

        elapsed = time.perf_counter() - start
        if elapsed > 3.0:
            print(f"⚠️ App creation slow: {elapsed:.2f}s")
        else:
            print(f"✅ App created successfully in {elapsed:.2f}s")

    except Exception as app_error:
        print(f"⚠️ App creation failed: {app_error}")


def check_file_access() -> None:
    """Check if credential files are accessible."""
    potential_paths = [
        ".env",
        "credentials/firebase-credentials.json",
        "credentials/gcp-credentials.json",
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
        os.getenv("FIREBASE_CREDENTIALS_PATH", ""),
    ]

    for path_str in potential_paths:
        if not path_str:
            continue

        path = Path(path_str)
        if path.exists() and path.is_file():
            # Check if file is accessible
            if os.access(path, os.R_OK):
                print(f"✅ Credential file accessible: {path}")
            else:
                print(f"⚠️ Credential file not readable: {path}")
        else:
            print(f"❌ Credential file not found: {path}")


async def main() -> None:
    """Main debug function."""
    print("🔍 CLARITY Startup Performance Debug")
    print("=" * 50)

    # Test imports
    print("\n📦 Testing imports...")
    slowest = find_slowest_import()
    if slowest:
        print(f"🐌 Slowest import: {slowest}")

    # Test config
    print("\n⚙️ Testing configuration...")
    test_config_access()

    # Test container
    print("\n🏭 Testing dependency container...")
    test_container_creation()

    # Test credentials
    print("\n🔐 Checking credential files...")
    check_file_access()

    # Test app creation
    print("\n🚀 Testing app creation...")
    await test_app_creation()

    print("\n✅ Debug complete!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback

        traceback.print_exc()
        print("\n💥 Debug failed - see traceback above")
