#!/usr/bin/env python3
"""
CLARITY Digital Twin Platform - Root Entry Point

Thin wrapper that launches the main FastAPI application from src/clarity/main.py.
This provides a convenient way to run the application from the project root
while maintaining proper package structure.

Usage:
    python main.py
    uvicorn main:app --reload
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and expose the FastAPI app from the proper location
from clarity.main import app

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
