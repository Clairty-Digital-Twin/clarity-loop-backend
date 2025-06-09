#!/usr/bin/env python3
"""DEFINITIVE Authentication Test Script
=====================================

This script definitively tests the authentication flow between iOS and backend.
It will establish the source of truth about whether auth is working.
"""

import base64
from datetime import UTC, datetime
import json
import time

import jwt  # PyJWT
import requests

# Modal production endpoint
BASE_URL = "https://crave-trinity-prod--clarity-backend-fastapi-app.modal.run"


def test_health_endpoint():
    """Test 1: Basic health check - no auth required"""
    print("\n🧪 TEST 1: Health Endpoint")
    print("=" * 50)

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200, "Health endpoint should return 200"
    assert response.json()["status"] == "healthy", "Service should be healthy"
    print("✅ PASSED: Health endpoint is working")
    return True


def test_protected_endpoint_without_auth():
    """Test 2: Access protected endpoint without authentication"""
    print("\n🧪 TEST 2: Protected Endpoint Without Auth")
    print("=" * 50)

    response = requests.get(f"{BASE_URL}/api/v1/health-data")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 401, "Should return 401 without auth"
    json_response = response.json()
    assert "detail" in json_response or "error" in json_response, "Should return error details"
    print("✅ PASSED: Protected endpoint correctly requires auth")
    return True


def test_debug_endpoint():
    """Test 3: Debug endpoint to check token verification"""
    print("\n🧪 TEST 3: Debug Token Verification Endpoint")
    print("=" * 50)

    # Test with a dummy token
    headers = {
        "Authorization": "Bearer dummy-token-12345",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/debug/verify-token-directly",
            headers=headers,
            timeout=30
        )
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            print(f"\nFirebase Admin Initialized: {data.get('firebase_admin_initialized', False)}")
            print(f"Auth Provider Type: {data.get('auth_provider_type', 'Unknown')}")
            print(f"Environment: {data.get('environment', {})}")
        else:
            print(f"Error Response: {response.text}")

    except Exception as e:
        print(f"❌ ERROR: {e!s}")
        return False

    return True


def decode_jwt_without_verification(token):
    """Decode JWT without verification to inspect claims"""
    try:
        # Decode without verification to see the contents
        decoded = jwt.decode(token, options={"verify_signature": False})
        return decoded
    except Exception as e:
        print(f"Error decoding JWT: {e}")
        return None


def test_with_real_token(token):
    """Test 4: Test with a real Firebase token"""
    print("\n🧪 TEST 4: Test With Real Firebase Token")
    print("=" * 50)

    # First decode the token to inspect it
    decoded = decode_jwt_without_verification(token)
    if decoded:
        print("\n📋 Token Claims:")
        print(f"  - User ID: {decoded.get('uid', 'N/A')}")
        print(f"  - Email: {decoded.get('email', 'N/A')}")
        print(f"  - Issued At: {datetime.fromtimestamp(decoded.get('iat', 0), UTC)}")
        print(f"  - Expires: {datetime.fromtimestamp(decoded.get('exp', 0), UTC)}")
        print(f"  - Audience: {decoded.get('aud', 'N/A')}")
        print(f"  - Issuer: {decoded.get('iss', 'N/A')}")

        # Check if token is expired
        if decoded.get('exp', 0) < time.time():
            print("⚠️  WARNING: Token is expired!")

    # Test the debug endpoint with the real token
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    print("\n🔍 Testing Debug Endpoint...")
    response = requests.post(
        f"{BASE_URL}/api/v1/debug/verify-token-directly",
        headers=headers,
        timeout=30
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Verification Success: {data.get('verification_success', False)}")
        if data.get('verification_success'):
            print("✅ TOKEN VERIFIED SUCCESSFULLY!")
            print(f"User Info: {json.dumps(data.get('user_info', {}), indent=2)}")
        else:
            print("❌ TOKEN VERIFICATION FAILED")
            print(f"Note: {data.get('note', 'No additional info')}")
    else:
        print(f"Error: {response.text}")

    # Now test a protected endpoint
    print("\n🔍 Testing Protected Endpoint...")
    response = requests.get(
        f"{BASE_URL}/api/v1/health-data",
        headers=headers,
        timeout=30
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ AUTHENTICATION SUCCESSFUL! Protected endpoint accessed!")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    print("❌ AUTHENTICATION FAILED")
    print(f"Error: {json.dumps(response.json(), indent=2)}")
    return False


def main():
    """Run all authentication tests"""
    print("🚀 DEFINITIVE AUTHENTICATION TEST SUITE")
    print("=" * 70)
    print(f"Target: {BASE_URL}")
    print(f"Time: {datetime.now(UTC).isoformat()}")
    print("=" * 70)

    # Run basic tests
    test_health_endpoint()
    test_protected_endpoint_without_auth()
    test_debug_endpoint()

    # Test with a real token if provided
    print("\n" + "=" * 70)
    print("📝 To test with a real Firebase token, run:")
    print("python test_auth_definitive.py YOUR_FIREBASE_TOKEN_HERE")
    print("=" * 70)

    # Check if token was provided as argument
    import sys
    if len(sys.argv) > 1:
        token = sys.argv[1]
        print(f"\n🔐 Testing with provided token (length: {len(token)})")
        success = test_with_real_token(token)

        print("\n" + "=" * 70)
        print("🏁 FINAL RESULT:")
        if success:
            print("✅ ✅ ✅ AUTHENTICATION IS WORKING! ✅ ✅ ✅")
            print("The backend successfully verified the Firebase token!")
        else:
            print("❌ ❌ ❌ AUTHENTICATION FAILED ❌ ❌ ❌")
            print("Check the logs above for specific errors")
        print("=" * 70)


if __name__ == "__main__":
    main()
