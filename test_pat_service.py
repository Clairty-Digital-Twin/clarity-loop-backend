#!/usr/bin/env python3
"""Test script to check PAT service status."""

import asyncio
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from clarity.ml.pat_service import get_pat_service

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_pat_service() -> None:
    """Test the PAT service loading and health."""
    try:
        logger.info("Testing PAT service...")
        service = await get_pat_service()
        logger.info("Service created: %s", service)

        logger.info("Loading model...")
        await service.load_model()
        logger.info("Model loaded: %s", service.is_loaded)

        logger.info("Checking health...")
        health = await service.health_check()
        logger.info("Health: %s", health)

    except (RuntimeError, ValueError, ConnectionError) as e:
        logger.exception("Error occurred during PAT service testing")


if __name__ == "__main__":
    asyncio.run(test_pat_service())
