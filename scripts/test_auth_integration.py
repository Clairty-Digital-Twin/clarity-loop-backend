#!/usr/bin/env python3
"""Integration test for authentication endpoint."""

from datetime import datetime
import json

import requests

# Test configuration
BASE_URL = "http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com"
API_ENDPOINT = f"{BASE_URL}/api/v1/auth/login"

# Test credentials (you'll need to create this user in Cognito)
test_payload = {
    "email": "test@example.com",
    "password": "TestPassword123!",
    "remember_me": True,
    "device_info": {
        "device_id": "test-device-123",
        "os_version": "macOS 14.0",
        "app_version": "1.0.0",
    },
}

print("Testing authentication endpoint...")
print(f"URL: {API_ENDPOINT}")
print(f"Payload: {json.dumps(test_payload, indent=2)}")
print("-" * 50)

try:
    # Make the request
    response = requests.post(
        API_ENDPOINT, json=test_payload, headers={"Content-Type": "application/json"}
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Response Body: {json.dumps(response.json(), indent=2)}")

    if response.status_code == 200:
        print("\n✅ SUCCESS! Authentication is working!")
        tokens = response.json()
        print(f"Access Token: {tokens.get('access_token', '')[:50]}...")
    else:
        print(f"\n❌ FAILED with status {response.status_code}")

except Exception as e:
    print(f"\n❌ Error making request: {e}")
