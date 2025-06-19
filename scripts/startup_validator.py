#!/usr/bin/env python3
"""CLARITY Startup Validator CLI.

Command-line tool for validating CLARITY startup configuration and services
without actually starting the application.
"""

from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from clarity.startup.orchestrator import main  # noqa: E402

if __name__ == "__main__":
    # Add default dry-run flag if not specified
    if not any(arg in sys.argv for arg in ["--dry-run", "--config-only", "--help"]):
        sys.argv.append("--dry-run")

    # Set permissions for script
    import stat

    script_path = Path(__file__)
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

    sys.exit(main())
