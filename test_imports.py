#!/usr/bin/env python3
"""Test script to validate that all critical imports work correctly.

This script helps debug module import issues in Docker or local environments.
"""

from pathlib import Path
import sys
import traceback


def test_imports() -> bool:
    """Test all critical imports for the application."""
    # Test 1: Basic Python path setup
    src_path = Path(__file__).parent / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
    else:
        return False

    # Test 2: Core clarity module
    try:
        import clarity  # type: ignore[import-untyped] # noqa: F401,PLC0415
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return False

    # Test 3: Main application modules
    try:
        from clarity.core.config import (  # noqa: PLC0415; type: ignore[import-untyped]
            get_settings,
        )

        _settings = get_settings()

        from clarity.core.container import (  # noqa: PLC0415; type: ignore[import-untyped]
            create_application,
        )

        _app_instance = create_application()

        from clarity.main import get_app  # type: ignore[import-untyped] # noqa: PLC0415
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return False

    # Test 4: FastAPI app creation
    try:
        # Add project root to path if not already there
        root_path = Path(__file__).parent
        if str(root_path) not in sys.path:
            sys.path.insert(0, str(root_path))

        from main import get_app  # noqa: PLC0415

        _app = get_app()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return False

    return True


def test_configuration() -> bool:
    """Test configuration loading."""
    try:
        from clarity.core.config import get_settings  # noqa: PLC0415

        _settings = get_settings()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return False
    else:
        return True


if __name__ == "__main__":

    success = True
    success &= test_imports()
    success &= test_configuration()

    if success:
        sys.exit(0)
    else:
        sys.exit(1)
