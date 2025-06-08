#!/usr/bin/env python3
"""Test authentication against Modal deployment."""

import asyncio
import logging
import os
from pathlib import Path
import sys

import httpx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_modal_auth():
    """Test authentication against Modal deployment."""
    # Modal URL - update this with your actual Modal URL
    base_url = os.getenv("MODAL_URL", "https://your-app--fastapi-app.modal.run")

    # Test cases
    test_cases = [
        {
            "name": "No token",
            "headers": {},
            "expected_status": [401, 403]
        },
        {
            "name": "Invalid token format",
            "headers": {"Authorization": "Bearer test-token-123"},
            "expected_status": [401]
        },
        {
            "name": "Empty bearer",
            "headers": {"Authorization": "Bearer "},
            "expected_status": [401]
        },
        {
            "name": "Wrong auth scheme",
            "headers": {"Authorization": "Basic dGVzdDp0ZXN0"},
            "expected_status": [401]
        }
    ]

    # If we have a real token from environment, test it
    real_token = os.getenv("FIREBASE_TOKEN")
    if real_token:
        test_cases.append({
            "name": "Real Firebase token",
            "headers": {"Authorization": f"Bearer {real_token}"},
            "expected_status": [200]
        })

    async with httpx.AsyncClient() as client:
        # Test health endpoint (should work without auth)
        logger.info("Testing health endpoint...")
        try:
            response = await client.get(f"{base_url}/health")
            logger.info(f"Health check: {response.status_code} - {response.json()}")
        except Exception as e:
            logger.error(f"Health check failed: {e}")

        # Test authenticated endpoints
        for test in test_cases:
            logger.info(f"\nTesting: {test['name']}")
            try:
                response = await client.get(
                    f"{base_url}/api/v1/auth/me",
                    headers=test['headers']
                )

                logger.info(f"Status: {response.status_code}")
                if response.status_code in test['expected_status']:
                    logger.info("✅ Test passed")
                else:
                    logger.error(f"❌ Test failed - expected {test['expected_status']}")

                try:
                    logger.info(f"Response: {response.json()}")
                except:
                    logger.info(f"Response text: {response.text[:200]}")

            except Exception as e:
                logger.error(f"Request failed: {e}")


if __name__ == "__main__":
    print("Modal Authentication Test")
    print("=" * 50)

    modal_url = input("Enter your Modal URL (e.g., https://yourapp--fastapi-app.modal.run): ").strip()
    if modal_url:
        os.environ["MODAL_URL"] = modal_url

    # Optional: test with a real token
    use_real_token = input("\nDo you have a Firebase token to test? (y/n): ").strip().lower()
    if use_real_token == 'y':
        token = input("Paste your Firebase token: ").strip()
        os.environ["FIREBASE_TOKEN"] = token

    asyncio.run(test_modal_auth())
