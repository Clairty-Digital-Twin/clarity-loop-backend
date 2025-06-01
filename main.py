#!/usr/bin/env python3
"""CLARITY Digital Twin Platform - Root Entry Point.

Thin wrapper that launches the main FastAPI application from src/clarity/main.py.
This provides a convenient way to run the application from the project root
while maintaining proper package structure.

Usage:
    python main.py
    uvicorn main:app --reload
"""

from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def get_app() -> "FastAPI":
    """Lazy import and return the FastAPI app to avoid circular imports."""
    from clarity.main import (
        get_application,  # type: ignore[import-untyped]
    )

    return get_application()  # type: ignore[no-any-return]


# Expose app for uvicorn
app = get_app()

if __name__ == "__main__":
    import uvicorn

    # Bind to localhost in development for security
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="info")
