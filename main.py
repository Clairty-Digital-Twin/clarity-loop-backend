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

# Add src directory to Python path (must be done before clarity imports)
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import clarity after path setup
from clarity.main import (  # noqa: E402
    get_app as clarity_get_app,  # type: ignore[import-untyped]
)

if TYPE_CHECKING:
    from fastapi import FastAPI


def get_app() -> "FastAPI":
    """Get the FastAPI app instance (clarity.main handles lazy initialization)."""
    return clarity_get_app()  # type: ignore[no-any-return]


# Expose app for uvicorn
app = get_app()

if __name__ == "__main__":
    import uvicorn

    # Bind to localhost in development for security
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="info")
