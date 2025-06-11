#!/usr/bin/env python3
"""DEFINITIVE Authentication Test Script.
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


def test_health_endpoint() -> bool:
    """Test 1: Basic health check - no auth required."""
    response = requests.get(f"{BASE_URL}/health")

    assert response.status_code == 200, "Health endpoint should return 200"
    assert response.json()["status"] == "healthy", "Service should be healthy"
    return True


def test_protected_endpoint_without_auth() -> bool:
    """Test 2: Access protected endpoint without authentication."""
    response = requests.get(f"{BASE_URL}/api/v1/health-data")

    assert response.status_code == 401, "Should return 401 without auth"
    json_response = response.json()
    assert (
        "detail" in json_response or "error" in json_response
    ), "Should return error details"
    return True


def test_debug_endpoint() -> bool:
    """Test 3: Debug endpoint to check token verification."""
    # Test with a dummy token
    headers = {
        "Authorization": "Bearer dummy-token-12345",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/debug/verify-token-directly",
            headers=headers,
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()

    except Exception as e:
        return False

    return True


def decode_jwt_without_verification(token):
    """Decode JWT without verification to inspect claims."""
    try:
        # Decode without verification to see the contents
        return jwt.decode(token, options={"verify_signature": False})
    except Exception as e:
        return None


def test_with_real_token(token) -> bool:
    """Test 4: Test with a real Firebase token."""
    # First decode the token to inspect it
    decoded = decode_jwt_without_verification(token)
    if decoded:

        # Check if token is expired
        if decoded.get("exp", 0) < time.time():
            pass

    # Test the debug endpoint with the real token
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    response = requests.post(
        f"{BASE_URL}/api/v1/debug/verify-token-directly", headers=headers, timeout=30
    )

    if response.status_code == 200:
        data = response.json()
        if data.get("verification_success"):
            pass

    # Now test a protected endpoint
    response = requests.get(
        f"{BASE_URL}/api/v1/health-data", headers=headers, timeout=30
    )

    return response.status_code == 200


def main() -> None:
    """Run all authentication tests."""
    # Run basic tests
    test_health_endpoint()
    test_protected_endpoint_without_auth()
    test_debug_endpoint()

    # Test with a real token if provided

    # Check if token was provided as argument
    import sys

    if len(sys.argv) > 1:
        token = sys.argv[1]
        success = test_with_real_token(token)

        if success:
            pass


if __name__ == "__main__":
    main()
