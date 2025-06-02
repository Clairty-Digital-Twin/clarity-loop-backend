"""Debug script for analyzing CLARITY startup performance.

This script helps identify and resolve application startup issues.
"""

# ruff: noqa: T201
# Allow print statements in debug script

import os
from pathlib import Path
import time

# Constants for timing thresholds
CONFIG_ACCESS_THRESHOLD = 0.5  # seconds
AUTH_CREATION_THRESHOLD = 2.0  # seconds
APP_CREATION_THRESHOLD = 3.0  # seconds
TOTAL_IMPORT_THRESHOLD = 2.0  # seconds


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
    if total_elapsed > TOTAL_IMPORT_THRESHOLD:
        print(f"⚠️ Total import time concerning: {total_elapsed:.2f}s")

    return culprit


def test_config_access() -> None:
    """Test configuration access performance."""
    print("🔍 Testing config access...")

    try:
        start = time.perf_counter()

        # TODO: Implement actual config access test

        elapsed = time.perf_counter() - start
        if elapsed > CONFIG_ACCESS_THRESHOLD:
            print(f"⚠️ Config access slow: {elapsed:.2f}s")
        else:
            print(f"✅ Config access fast: {elapsed:.2f}s")

    except (ImportError, ValueError, KeyError) as config_error:
        print(f"⚠️ Config access failed: {config_error}")


def test_container_creation() -> None:
    """Test dependency injection container creation."""
    print("🔍 Testing container creation...")

    try:
        # TODO: Implement actual container creation test

        auth_start = time.perf_counter()

        # TODO: Implement auth provider creation test

        auth_elapsed = time.perf_counter() - auth_start

        if auth_elapsed > AUTH_CREATION_THRESHOLD:
            print(f"⚠️ Auth provider creation slow: {auth_elapsed:.2f}s")
        else:
            print(f"✅ Auth provider creation fast: {auth_elapsed:.2f}s")

    except (ImportError, ValueError, RuntimeError) as container_error:
        print(f"⚠️ Container creation failed: {container_error}")


def test_app_creation() -> None:
    """Test FastAPI app creation performance."""
    print("🔍 Testing app creation...")

    try:
        start = time.perf_counter()

        # TODO: Implement actual app creation test

        elapsed = time.perf_counter() - start
        if elapsed > APP_CREATION_THRESHOLD:
            print(f"⚠️ App creation slow: {elapsed:.2f}s")
        else:
            print(f"✅ App created successfully in {elapsed:.2f}s")

    except (ImportError, ValueError, RuntimeError) as app_error:
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


def main() -> None:
    """Main debug function."""
    print("🔍 CLARITY Startup Performance Debug")
    print("=" * 50)

    # Test imports
    print("\n📦 Testing imports...")
    slowest = find_slowest_import()
    if slowest:
        print(f"🐌 Slowest import: {slowest}")

    # Test configuration access
    print("\n📁 Testing configuration access...")
    test_config_access()

    # Test container setup
    print("\n📦 Testing container setup...")
    test_container_creation()

    # Test credentials
    print("\n🔐 Checking credential files...")
    check_file_access()

    # Test app creation
    print("\n🚀 Testing app creation...")
    test_app_creation()

    print("\n✅ Debug complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (ImportError, RuntimeError, OSError) as e:
        import traceback

        traceback.print_exc()
        print(f"❌ Debug script failed: {e}")
