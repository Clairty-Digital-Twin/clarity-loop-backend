#!/usr/bin/env python3
"""Test script to verify AWS deployment configuration."""

import asyncio
import logging
import os
from typing import Dict, Any

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
    
    settings = get_settings()
    
    print(f"Environment: {settings.environment}")
    print(f"AWS Region: {settings.aws_region}")
    print(f"Skip External Services: {settings.skip_external_services}")
    print(f"DynamoDB Table: {settings.dynamodb_table_name}")
    print(f"S3 Bucket: {settings.s3_bucket_name}")
    print(f"Gemini Model: {settings.gemini_model}")
    
    return settings


async def test_container_initialization():
    """Test dependency container initialization."""
    logger.info("Testing container initialization...")
    
    container = DependencyContainer()
    await container.initialize()
    
    # Check services
    print(f"Auth Provider: {type(container.auth_provider).__name__}")
    print(f"Repository: {type(container.health_data_repository).__name__}")
    print(f"Gemini Service: {'Configured' if container.gemini_service else 'Not configured'}")
    
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
    print(f"Root endpoint status: {response.status_code}")
    print(f"Root response: {response.json()}")
    
    # Test health endpoint
    response = client.get("/health")
    print(f"Health endpoint status: {response.status_code}")
    print(f"Health response: {response.json()}")
    
    return response.json()


async def main():
    """Run all tests."""
    print("=" * 50)
    print("CLARITY AWS Deployment Test")
    print("=" * 50)
    
    try:
        # Test configuration
        print("\n1. Configuration Test:")
        await test_configuration()
        
        # Test container
        print("\n2. Container Initialization Test:")
        await test_container_initialization()
        
        # Test API
        print("\n3. API Endpoints Test:")
        await test_api_endpoints()
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())