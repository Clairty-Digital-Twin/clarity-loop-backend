#!/usr/bin/env python3
"""Test backend's ability to handle various JSON formats."""

import asyncio
import json

import httpx

BASE_URL = "http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com"

# Test cases with different JSON formats
test_cases = [
    {
        "name": "Standard JSON (works)",
        "payload": {
            "email": "test@example.com",
            "password": "TestPassword123!",
            "remember_me": True,
            "device_info": {
                "device_id": "test-device",
                "os_version": "iOS 18.0",
                "app_version": "1.0.0",
            },
        },
    },
    {
        "name": "JSON with extra whitespace",
        "raw": '{\n  "email" : "test@example.com",\n  "password" : "TestPassword123!",\n  "remember_me" : true,\n  "device_info" : {\n    "device_id" : "test",\n    "os_version" : "iOS 18.0",\n    "app_version" : "1.0.0"\n  }\n}',
    },
    {
        "name": "Compact JSON (no spaces)",
        "raw": '{"email":"test@example.com","password":"TestPassword123!","remember_me":true,"device_info":{"device_id":"test","os_version":"iOS 18.0","app_version":"1.0.0"}}',
    },
    {
        "name": "JSON with Unicode characters",
        "payload": {
            "email": "test@example.com",
            "password": "TestPassword123!",
            "remember_me": True,
            "device_info": {
                "device_id": "iPhone™",
                "os_version": "iOS 18.0",
                "app_version": "1.0.0",
            },
        },
    },
    {
        "name": "JSON with escaped quotes",
        "raw": '{"email":"test@example.com","password":"TestPassword123!","remember_me":true,"device_info":{"device_id":"test\\"device","os_version":"iOS 18.0","app_version":"1.0.0"}}',
    },
]


async def test_json_handling() -> None:
    """Test backend JSON handling."""
    async with httpx.AsyncClient() as client:
        for test in test_cases:
            # Prepare request data
            if "raw" in test:
                # Use raw JSON string
                data = test["raw"]
                headers = {"Content-Type": "application/json"}
            else:
                # Use payload dict
                data = json.dumps(test["payload"])
                headers = {"Content-Type": "application/json"}

            try:
                # Send to login endpoint
                response = await client.post(
                    f"{BASE_URL}/api/v1/auth/login", content=data, headers=headers
                )

                if response.status_code == 200:
                    print(f"✓ {test['name']}: Success")
                else:
                    print(f"✗ {test['name']}: Failed with status {response.status_code}")

            except (httpx.HTTPError, httpx.ConnectError, ValueError, json.JSONDecodeError) as e:
                # Log JSON handling errors for debugging
                print(f"JSON handling error with test '{test.get('name', 'unknown')}': {e}")


async def test_debug_endpoint() -> None:
    """Test the debug endpoint to see raw request details."""
    # Create a problematic payload similar to what iOS might send
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

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/debug/capture-raw-request", json=test_payload
        )

        if response.status_code == 200:
            data = response.json()
            print(f"Debug endpoint response: {json.dumps(data, indent=2)}")
        else:
            print(f"Debug endpoint failed with status {response.status_code}")


if __name__ == "__main__":
    asyncio.run(test_json_handling())
    asyncio.run(test_debug_endpoint())
