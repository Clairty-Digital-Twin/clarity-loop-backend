#!/usr/bin/env python3
"""Production readiness test for AWS deployed backend."""

from datetime import UTC, datetime
import json
import sys
import time

import requests

BASE_URL = "http://***REMOVED***"
API_KEY = "production-api-key-change-me"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def test_health_check() -> None:
    """Test basic health endpoint."""
    resp = requests.get(f"{BASE_URL}/health", timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["features"]["cognito_auth"] is True
    assert data["features"]["gemini_insights"] is True


def test_auth_signup():
    """Test user signup via Cognito."""
    test_email = f"test_{int(time.time())}@clarity.health"
    signup_data = {"email": test_email, "password": "TestPassword123!"}

    resp = requests.post(
        f"{BASE_URL}/api/v1/auth/signup", headers=HEADERS, json=signup_data, timeout=10
    )

    # Should return 200 even if Cognito is not fully configured
    assert resp.status_code == 200
    data = resp.json()

    if data.get("success"):
        return test_email, signup_data["password"], data.get("tokens")
    return test_email, signup_data["password"], None


def test_auth_login(email, password):
    """Test user login."""
    login_data = {"email": email, "password": password}

    resp = requests.post(
        f"{BASE_URL}/api/v1/auth/login", headers=HEADERS, json=login_data, timeout=10
    )

    assert resp.status_code == 200
    data = resp.json()

    if data.get("success") and data.get("tokens"):
        return data["tokens"]["id_token"]
    return None


def test_health_data_with_api_key():
    """Test health data storage with API key."""
    health_data = {
        "data_type": "heart_rate",
        "value": 72.5,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    resp = requests.post(
        f"{BASE_URL}/api/v1/health-data", headers=HEADERS, json=health_data, timeout=10
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "data_id" in data
    return data["data_id"]


def test_health_data_query() -> None:
    """Test health data retrieval."""
    resp = requests.get(
        f"{BASE_URL}/api/v1/health-data?limit=5", headers=HEADERS, timeout=10
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "data" in data


def test_insights_generation() -> None:
    """Test Gemini insights."""
    insight_request = {
        "query": "What are some tips for better sleep?",
        "include_recent_data": False,
    }

    resp = requests.post(
        f"{BASE_URL}/api/v1/insights", headers=HEADERS, json=insight_request, timeout=30
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "insight" in data


def test_user_profile_api_key() -> None:
    """Test user profile with API key."""
    resp = requests.get(f"{BASE_URL}/api/v1/user/profile", headers=HEADERS, timeout=10)

    assert resp.status_code == 200
    data = resp.json()
    assert data["auth_type"] == "api_key"


def test_openapi_docs() -> None:
    """Test API documentation."""
    resp = requests.get(f"{BASE_URL}/docs", timeout=5)
    assert resp.status_code == 200

    resp = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
    assert resp.status_code == 200
    openapi = resp.json()
    assert openapi["info"]["title"] == "Clarity Health Backend (AWS Full)"


def run_all_tests() -> bool | None:
    """Run all production readiness tests."""
    try:
        # Basic health check
        test_health_check()

        # API documentation
        test_openapi_docs()

        # Authentication flow
        email, password, tokens = test_auth_signup()
        auth_token = (
            test_auth_login(email, password) if not tokens else tokens.get("id_token")
        )

        # Data operations with API key
        test_health_data_with_api_key()
        test_health_data_query()

        # AI insights
        test_insights_generation()

        # User profile
        test_user_profile_api_key()

        # If we have auth token, test authenticated endpoints
        if auth_token:
            auth_headers = HEADERS.copy()
            auth_headers["Authorization"] = f"Bearer {auth_token}"

            # Test with JWT auth
            resp = requests.get(
                f"{BASE_URL}/api/v1/user/profile", headers=auth_headers, timeout=10
            )
            if resp.status_code == 200:
                pass

        return True

    except AssertionError as e:
        return False
    except Exception as e:
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
