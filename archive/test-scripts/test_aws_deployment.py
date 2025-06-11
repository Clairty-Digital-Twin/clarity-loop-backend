#!/usr/bin/env python3
"""Test script to verify AWS deployment configuration."""

import asyncio
import logging
import os
from typing import Any, Dict

# Set environment variables for testing
os.environ["ENVIRONMENT"] = "development"
os.environ["SKIP_EXTERNAL_SERVICES"] = "true"
os.environ["GEMINI_API_KEY"] = "test-key"

from clarity.core.config_aws import get_settings
from clarity.core.container_aws import DependencyContainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_configuration():
    """Test AWS configuration loading."""
    logger.info("Testing AWS configuration...")

    return get_settings()


async def test_container_initialization():
    """Test dependency container initialization."""
    logger.info("Testing container initialization...")

    container = DependencyContainer()
    await container.initialize()

    # Check services

    await container.shutdown()

    return container


async def test_api_endpoints():
    """Test basic API endpoints."""
    logger.info("Testing API endpoints...")

    from fastapi.testclient import TestClient

    from clarity.main_aws import app

    client = TestClient(app)

    # Test root endpoint
    response = client.get("/")

    # Test health endpoint
    response = client.get("/health")

    return response.json()


async def main() -> None:
    """Run all tests."""
    try:
        # Test configuration
        await test_configuration()

        # Test container
        await test_container_initialization()

        # Test API
        await test_api_endpoints()

    except Exception as e:
        raise


if __name__ == "__main__":
    asyncio.run(main())
