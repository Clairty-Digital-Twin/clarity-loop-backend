#!/usr/bin/env python3
"""Test script to validate that all critical imports work correctly.

This script helps debug module import issues in Docker or local environments.
"""

from pathlib import Path
import sys
import traceback


def test_imports() -> bool:
    """Test all critical imports for the application."""
    print("🔍 Testing module imports...")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path}")

    # Test 1: Basic Python path setup
    print("\n1. Testing Python path setup...")
    src_path = Path(__file__).parent / "src"
    if src_path.exists():
        print(f"✅ src directory exists: {src_path}")
        sys.path.insert(0, str(src_path))
        print(f"✅ Added to Python path: {src_path}")
    else:
        print(f"❌ src directory not found: {src_path}")
        return False

    # Test 2: Core clarity module
    print("\n2. Testing clarity module import...")
    try:
        import clarity  # type: ignore[import-untyped]
        print(f"✅ clarity module imported: {clarity.__file__}")
        print(f"   Version: {clarity.__version__}")
    except Exception as e:
        print(f"❌ Failed to import clarity: {e}")
        traceback.print_exc()
        return False

        # Test 3: Main application modules
    print("\n3. Testing core application modules...")
    try:
        from clarity.core.config import get_settings  # type: ignore[import-untyped]
        settings = get_settings()
        print(f"✅ clarity.core.config imported (environment: {settings.environment})")

        from clarity.core.container import (
            create_application,  # type: ignore[import-untyped]
        )
        app_instance = create_application()
        print(f"✅ clarity.core.container imported (app: {app_instance.title})")

        from clarity.main import get_app  # type: ignore[import-untyped]
        print("✅ clarity.main imported")
    except Exception as e:
        print(f"❌ Failed to import core modules: {e}")
        traceback.print_exc()
        return False

    # Test 4: Root main.py import
    print("\n4. Testing root main.py import...")
    try:
        # Add project root to path if not already there
        root_path = Path(__file__).parent
        if str(root_path) not in sys.path:
            sys.path.insert(0, str(root_path))

        import main
        print(f"✅ main module imported: {main.__file__}")
        print(f"   App instance: {main.app}")
    except Exception as e:
        print(f"❌ Failed to import main: {e}")
        traceback.print_exc()
        return False

    # Test 5: FastAPI app creation
    print("\n5. Testing FastAPI app creation...")
    try:
        from main import get_app
        app = get_app()
        print(f"✅ FastAPI app created: {app}")
        print(f"   App title: {app.title}")
        print(f"   App version: {app.version}")
    except Exception as e:
        print(f"❌ Failed to create FastAPI app: {e}")
        traceback.print_exc()
        return False

    print("\n🎉 All imports successful!")
    return True


def test_configuration() -> bool:
    """Test configuration loading."""
    print("\n📋 Testing configuration...")
    try:
        from clarity.core.config import get_settings
        settings = get_settings()
        print("✅ Settings loaded")
        print(f"   Environment: {settings.environment}")
        print(f"   Debug: {settings.debug}")
        print(f"   Host: {settings.host}")
        print(f"   Port: {settings.port}")
        return True
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Starting import validation test...\n")

    success = True
    success &= test_imports()
    success &= test_configuration()

    if success:
        print("\n✅ All tests passed! The application should start successfully.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)
