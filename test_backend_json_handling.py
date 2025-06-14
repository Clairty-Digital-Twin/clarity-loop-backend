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


async def test_json_handling():
    """Test backend JSON handling."""
    print("=" * 60)
    print("BACKEND JSON HANDLING TEST")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        for i, test in enumerate(test_cases):
            print(f"\nTest {i + 1}: {test['name']}")
            print("-" * 40)

            # Prepare request data
            if "raw" in test:
                # Use raw JSON string
                data = test["raw"]
                headers = {"Content-Type": "application/json"}
                print(f"Raw JSON: {data}")
            else:
                # Use payload dict
                data = json.dumps(test["payload"])
                headers = {"Content-Type": "application/json"}
                print(f"JSON: {data}")

            try:
                # Send to login endpoint
                response = await client.post(
                    f"{BASE_URL}/api/v1/auth/login", content=data, headers=headers
                )

                print(f"Status: {response.status_code}")
                if response.status_code == 200:
                    print("✅ SUCCESS")
                else:
                    print(f"❌ FAILED: {response.text[:200]}")

            except Exception as e:
                print(f"❌ ERROR: {e}")


async def test_debug_endpoint():
    """Test the debug endpoint to see raw request details."""
    print("\n" + "=" * 60)
    print("DEBUG ENDPOINT TEST")
    print("=" * 60)

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
            print("Debug Response:")
            print(json.dumps(data, indent=2))
        else:
            print(f"Debug endpoint failed: {response.status_code}")


if __name__ == "__main__":
    asyncio.run(test_json_handling())
    asyncio.run(test_debug_endpoint())
