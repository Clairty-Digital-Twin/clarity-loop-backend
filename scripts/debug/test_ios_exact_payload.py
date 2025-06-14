#!/usr/bin/env python3
"""Test with exact iOS payload structure."""

import asyncio
import json

import httpx

BASE_URL = "http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com"

# This is EXACTLY what iOS sends based on the Swift code
ios_payload = {
    "email": "test@example.com",
    "password": "TestPassword123!",
    "remember_me": True,  # Snake case from JSONEncoder
    "device_info": {  # Snake case from JSONEncoder
        "device_id": "iPhone-123",
        "os_version": "iOS 18.0",
        "app_version": "1.0.0",
    },
}


async def test_ios_payload() -> None:
    """Test with exact iOS payload."""
    # Test 1: Standard JSON encoding
    json_str = json.dumps(ios_payload)

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/api/v1/auth/login", json=ios_payload)
        if response.status_code != 200:
            pass

    # Test 2: Pretty printed JSON (iOS might do this)
    json_pretty = json.dumps(ios_payload, indent=2)
    if len(json_pretty) > 55:
        pass

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/auth/login",
            content=json_pretty,
            headers={"Content-Type": "application/json"},
        )
        if response.status_code != 200:
            pass

    # Test 3: With potential escape issues
    # The error mentions "Invalid \escape" at position 55
    # Let's see what's at position 55 in different formats

    test_strings = [
        json.dumps(ios_payload),
        json.dumps(ios_payload, separators=(",", ":")),  # Compact
        json.dumps(ios_payload, indent=2),  # Pretty
        json.dumps(ios_payload, indent=4),  # More indent
    ]

    for test_str in test_strings:
        if len(test_str) > 55:
            pass


if __name__ == "__main__":
    asyncio.run(test_ios_payload())
