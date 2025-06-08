#!/usr/bin/env python3
"""Test authentication flow with the fixed middleware."""

import asyncio
import logging
from datetime import datetime, timedelta, UTC

from fastapi.testclient import TestClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_auth_flow():
    """Test the authentication flow with both debug and insights endpoints."""
    # Import after setting up logging
    from src.clarity.main import app
    
    # Create test client
    client = TestClient(app)
    
    # Create a mock token (this would normally come from Firebase)
    mock_token = "eyJhbGciOiJSUzI1NiIsImtpZCI6InRlc3QifQ.eyJ1aWQiOiJ0ZXN0MTIzIiwiZW1haWwiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImlhdCI6MTczMTAwMDAwMCwiZXhwIjoxNzMxMDAzNjAwfQ.test_signature"
    
    headers = {"Authorization": f"Bearer {mock_token}"}
    
    # Test 1: Health endpoint (no auth required)
    logger.info("\n=== Testing /health endpoint (no auth) ===")
    response = client.get("/health")
    logger.info(f"Status: {response.status_code}")
    logger.info(f"Response: {response.json()}")
    
    # Test 2: Debug auth-check endpoint
    logger.info("\n=== Testing /api/v1/debug/auth-check endpoint ===")
    response = client.get("/api/v1/debug/auth-check", headers=headers)
    logger.info(f"Status: {response.status_code}")
    logger.info(f"Response: {response.json() if response.status_code != 401 else response.text}")
    
    # Test 3: Insights status endpoint
    logger.info("\n=== Testing /api/v1/insights/status endpoint ===")
    response = client.get("/api/v1/insights/status", headers=headers)
    logger.info(f"Status: {response.status_code}")
    logger.info(f"Response: {response.json() if response.status_code != 401 else response.text}")
    
    # Test 4: Insights generate endpoint
    logger.info("\n=== Testing /api/v1/insights/generate endpoint ===")
    insight_data = {
        "analysis_results": {"test": "data"},
        "context": "Test context",
        "insight_type": "brief"
    }
    response = client.post("/api/v1/insights/generate", json=insight_data, headers=headers)
    logger.info(f"Status: {response.status_code}")
    logger.info(f"Response: {response.json() if response.status_code != 401 else response.text}")

if __name__ == "__main__":
    asyncio.run(test_auth_flow())