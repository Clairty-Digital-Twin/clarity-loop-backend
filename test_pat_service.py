#!/usr/bin/env python3
"""Test script to check PAT service status."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from clarity.ml.pat_service import get_pat_service


async def test_pat_service():
    """Test the PAT service loading and health."""
    try:
        print("Testing PAT service...")
        service = await get_pat_service()
        print(f"Service created: {service}")
        
        print("Loading model...")
        await service.load_model()
        print(f"Model loaded: {service.is_loaded}")
        
        print("Checking health...")
        health = await service.health_check()
        print(f"Health: {health}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_pat_service()) 