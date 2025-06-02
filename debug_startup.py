"""CLARITY Digital Twin Platform - Startup Diagnostic Tool

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
    print("🔍 DIAGNOSING STARTUP HANG...")
    print("=" * 50)

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
        print(f"\n⏱️  Testing {name}...")
        start = time.perf_counter()

        try:
            await asyncio.wait_for(test_func(), timeout=10.0)
            elapsed = time.perf_counter() - start
            print(f"✅ {name}: {elapsed:.2f}s")

        except TimeoutError:
            elapsed = time.perf_counter() - start
            print(f"🚨 {name}: HANGS (>{elapsed:.1f}s) - THIS IS THE CULPRIT")
            culprit = name
            break

        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"💥 {name}: ERROR after {elapsed:.2f}s - {type(e).__name__}: {e}")
            culprit = name
            break

    total_elapsed = time.perf_counter() - overall_start
    print(f"\n🏁 Total diagnostic time: {total_elapsed:.2f}s")

    return culprit


async def setup_logging_test() -> None:
    """Test logging setup component."""
    from clarity.core.logging_config import setup_logging

    setup_logging()
    print("   📝 Logging configured successfully")


async def test_config_provider() -> None:
    """Test config provider initialization."""
    from clarity.core.container import get_container

    container = get_container()
    config = container.get_config_provider()

    # Test some config access
    is_dev = config.is_development()
    auth_enabled = config.is_auth_enabled()
    firebase_config = config.get_firebase_config()

    print(f"   ⚙️  Config loaded - Dev: {is_dev}, Auth: {auth_enabled}")
    print(f"   🔧 Firebase project: {firebase_config.get('project_id', 'Not set')}")


async def test_auth_provider_init() -> None:
    """Test auth provider initialization."""
    from clarity.core.container import get_container

    container = get_container()

    print("   🔐 Getting auth provider...")
    auth = container.get_auth_provider()
    print(f"   🔐 Auth provider type: {type(auth).__name__}")

    if hasattr(auth, "initialize"):
        print("   🔐 Calling auth.initialize()...")
        await auth.initialize()
        print("   🔐 Auth provider initialized")
    else:
        print("   🔐 Auth provider has no initialize() method")


async def test_repository_init() -> None:
    """Test repository initialization."""
    from clarity.core.container import get_container

    container = get_container()

    print("   🗄️  Getting repository...")
    repo = container.get_health_data_repository()
    print(f"   🗄️  Repository type: {type(repo).__name__}")

    if hasattr(repo, "initialize"):
        print("   🗄️  Calling repo.initialize()...")
        await repo.initialize()
        print("   🗄️  Repository initialized")
    else:
        print("   🗄️  Repository has no initialize() method")


async def test_full_lifespan() -> None:
    """Test the full lifespan context manager."""
    from clarity.core.container import get_container

    container = get_container()

    print("   🚀 Testing full lifespan simulation...")

    # Simulate what the lifespan does
    from clarity.core.logging_config import setup_logging

    setup_logging()
    print("   📝 Lifespan: Logging setup complete")

    # Initialize auth provider
    auth_provider = container.get_auth_provider()
    if hasattr(auth_provider, "initialize"):
        await auth_provider.initialize()
    print("   🔐 Lifespan: Auth provider initialized")

    # Initialize repository
    repository = container.get_health_data_repository()
    if hasattr(repository, "initialize"):
        await repository.initialize()
    print("   🗄️  Lifespan: Repository initialized")

    print("   🚀 Full lifespan simulation completed successfully")


async def test_environment_variables() -> None:
    """Test environment variable availability."""
    import os

    print("\n🌍 ENVIRONMENT VARIABLES CHECK:")
    print("-" * 30)

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
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")


async def test_file_permissions() -> None:
    """Test file access permissions."""
    import os
    from pathlib import Path

    print("\n📁 FILE PERMISSIONS CHECK:")
    print("-" * 25)

    # Check common credential file locations
    credential_paths = [
        "credentials/firebase-credentials.json",
        "credentials/service-account.json",
        ".env",
        "src/clarity",
    ]

    for path_str in credential_paths:
        path = Path(path_str)
        if path.exists():
            if path.is_file():
                readable = os.access(path, os.R_OK)
                print(
                    f"{'✅' if readable else '❌'} {path}: {'Readable' if readable else 'Not readable'}"
                )
            else:
                print(f"📁 {path}: Directory exists")
        else:
            print(f"❌ {path}: Does not exist")


if __name__ == "__main__":
    print("🔥 CLARITY STARTUP DIAGNOSTICS")
    print("=" * 50)

    try:
        # First check environment and files
        asyncio.run(test_environment_variables())
        asyncio.run(test_file_permissions())

        print(f"\n{'=' * 50}")
        print("🧪 COMPONENT TESTING")
        print("=" * 50)

        # Main diagnostic
        culprit = asyncio.run(debug_startup())

        print(f"\n{'=' * 50}")
        print("📊 DIAGNOSIS RESULTS")
        print("=" * 50)

        if culprit:
            print(f"🎯 FOUND THE PROBLEM: {culprit}")
            print("\n💡 RECOMMENDED ACTIONS:")

            if "Auth provider" in culprit:
                print("   • Check Firebase credentials and project configuration")
                print("   • Verify FIREBASE_PROJECT_ID and FIREBASE_CREDENTIALS_PATH")
                print("   • Consider disabling auth in development (ENABLE_AUTH=false)")

            elif "Repository" in culprit:
                print("   • Check Firestore/Firebase credentials")
                print("   • Verify GCP_PROJECT_ID configuration")
                print("   • Consider using mock repository in development")

            elif "Config provider" in culprit:
                print("   • Check environment variable setup")
                print("   • Verify .env file exists and is readable")

            elif "Logging" in culprit:
                print("   • Check logging configuration and permissions")

            else:
                print("   • Review the specific error message above")
                print("   • Check network connectivity to Google services")

        else:
            print("✅ All components start fine - issue may be elsewhere")
            print("💡 Possible causes:")
            print("   • Race condition in lifespan execution")
            print("   • Event loop issues")
            print("   • FastAPI lifespan context manager problems")

    except KeyboardInterrupt:
        print("\n🛑 Diagnostics interrupted by user")
    except Exception as e:
        print(f"\n💥 Diagnostic script failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
