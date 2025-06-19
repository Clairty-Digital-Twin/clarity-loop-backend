#!/usr/bin/env python3
"""Simple script to test rate limiting functionality.

This demonstrates the rate limiting middleware in action.
"""

import time

import requests

# Constants
RATE_LIMIT_STATUS_CODE = 429


def test_rate_limiting(base_url: str | None = None) -> None:  # noqa: PT028
    if base_url is None:
        base_url = "http://localhost:8000"
    """Test rate limiting on various endpoints."""
    print("🔥 Rate Limiting Test Script")
    print("=" * 50)

    # Test login endpoint (10 requests/minute limit)
    print("\n1. Testing login endpoint rate limit (10/minute)...")
    login_url = f"{base_url}/api/v1/auth/login"

    for i in range(12):
        try:
            response = requests.post(
                login_url,
                json={"email": "test@example.com", "password": "wrongpass"},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

            print(f"   Request {i + 1}: Status {response.status_code}")

            # Check rate limit headers
            if "X-RateLimit-Limit" in response.headers:
                limit = response.headers["X-RateLimit-Limit"]
                remaining = response.headers.get("X-RateLimit-Remaining", "?")
                print(f"   Rate Limit: {remaining}/{limit} remaining")

            if response.status_code == RATE_LIMIT_STATUS_CODE:
                print(f"   ❌ Rate limit exceeded: {response.json()}")
                break

        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"   Error: {e}")

        # Small delay between requests
        time.sleep(0.5)

    print("\n2. Testing health check endpoint (no rate limit)...")
    health_url = f"{base_url}/api/v1/health"

    for i in range(5):
        try:
            response = requests.get(health_url, timeout=10)
            print(f"   Request {i + 1}: Status {response.status_code}")
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"   Error: {e}")

    print("\n✅ Rate limiting test complete!")


def test_rate_limit_headers(base_url: str | None = None) -> None:  # noqa: PT028
    if base_url is None:
        base_url = "http://localhost:8000"
    """Test rate limit headers in responses."""
    print("\n3. Checking rate limit headers...")

    # Make a request to a rate-limited endpoint
    response = requests.get(f"{base_url}/api/v1/health-data/health", timeout=10)

    print(f"   Status: {response.status_code}")
    print("   Headers:")
    for header in ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]:
        if header in response.headers:
            print(f"   - {header}: {response.headers[header]}")


if __name__ == "__main__":
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    print(f"Testing against: {base_url}")

    try:
        test_rate_limiting(base_url)
        test_rate_limit_headers(base_url)
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to server. Is it running?")
        print("   Start the server with: uvicorn src.clarity.main:app --reload")
        sys.exit(1)
