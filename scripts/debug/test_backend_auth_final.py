#!/usr/bin/env python3
"""Final comprehensive test of backend authentication."""

import asyncio

import httpx

# Test configuration
BASE_URL = "http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com"
LOGIN_ENDPOINT = f"{BASE_URL}/api/v1/auth/login"

# Test payload exactly as frontend sends it
test_payload = {
    "email": "test@example.com",
    "password": "TestPassword123!",
    "remember_me": True,
    "device_info": {
        "device_id": "iPhone-123",
        "os_version": "iOS 18.0",
        "app_version": "1.0.0",
    },
}


async def test_authentication() -> None:
    """Test authentication endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            # Make the request
            response = await client.post(
                LOGIN_ENDPOINT,
                json=test_payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                data = response.json()
                print(f"Authentication successful: {data}")
            else:
                print(f"Authentication failed with status {response.status_code}: {response.text}")

        except Exception as e:
            # Log authentication errors for debugging
            print(f"Authentication error: {e}")


if __name__ == "__main__":
    asyncio.run(test_authentication())
