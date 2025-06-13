#!/usr/bin/env python3
"""Test with exact iOS payload structure."""

import httpx
import json
import asyncio

BASE_URL = "http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com"

# This is EXACTLY what iOS sends based on the Swift code
ios_payload = {
    "email": "test@example.com",
    "password": "TestPassword123!",
    "remember_me": True,  # Snake case from JSONEncoder
    "device_info": {      # Snake case from JSONEncoder
        "device_id": "iPhone-123",
        "os_version": "iOS 18.0", 
        "app_version": "1.0.0"
    }
}

async def test_ios_payload():
    """Test with exact iOS payload."""
    print("=" * 60)
    print("iOS EXACT PAYLOAD TEST")
    print("=" * 60)
    
    # Test 1: Standard JSON encoding
    print("\nTest 1: Standard JSON")
    print("-" * 40)
    json_str = json.dumps(ios_payload)
    print(f"JSON: {json_str}")
    print(f"Length: {len(json_str)} bytes")
    print(f"Char at position 55: '{json_str[55] if len(json_str) > 55 else 'N/A'}'")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/auth/login",
            json=ios_payload
        )
        print(f"Status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response: {response.text}")
    
    # Test 2: Pretty printed JSON (iOS might do this)
    print("\n\nTest 2: Pretty Printed JSON")
    print("-" * 40)
    json_pretty = json.dumps(ios_payload, indent=2)
    print(f"JSON:\n{json_pretty}")
    print(f"Length: {len(json_pretty)} bytes")
    if len(json_pretty) > 55:
        print(f"Char at position 55: '{json_pretty[55]}' (ASCII: {ord(json_pretty[55])})")
        print(f"Context around 55: ...{json_pretty[50:60]}...")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/auth/login",
            content=json_pretty,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response: {response.text}")
    
    # Test 3: With potential escape issues
    print("\n\nTest 3: Testing Position 55 Issue")
    print("-" * 40)
    # The error mentions "Invalid \escape" at position 55
    # Let's see what's at position 55 in different formats
    
    test_strings = [
        json.dumps(ios_payload),
        json.dumps(ios_payload, separators=(',', ':')),  # Compact
        json.dumps(ios_payload, indent=2),              # Pretty
        json.dumps(ios_payload, indent=4),              # More indent
    ]
    
    for i, test_str in enumerate(test_strings):
        print(f"\nFormat {i+1}:")
        print(f"Length: {len(test_str)}")
        if len(test_str) > 55:
            print(f"Position 55: '{test_str[55]}' (ASCII: {ord(test_str[55])})")
            print(f"Context 50-60: '{test_str[50:60]}'")
            print(f"Context 45-65: '{test_str[45:65]}'")

if __name__ == "__main__":
    asyncio.run(test_ios_payload())