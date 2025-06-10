#!/usr/bin/env python3
"""Test deployment endpoints with Firebase auth token."""

import json
import sys

import requests

# Your Firebase token
AUTH_TOKEN = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImE0YTEwZGVjZTk4MzY2ZDZmNjNlMTY3Mjg2YWU5YjYxMWQyYmFhMjciLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL3NlY3VyZXRva2VuLmdvb2dsZS5jb20vY2xhcml0eS1sb29wLWJhY2tlbmQiLCJhdWQiOiJjbGFyaXR5LWxvb3AtYmFja2VuZCIsImF1dGhfdGltZSI6MTc0OTMzMjQ4OCwidXNlcl9pZCI6InZXNmZWajZreFdnem5rU2hXUzZSNEZXRWg0SjIiLCJzdWIiOiJ2VzZmVmo2a3hXZ3pua1NoV1M2UjRGV0VoNEoyIiwiaWF0IjoxNzQ5NDkyMDQyLCJleHAiOjE3NDk0OTU2NDIsImVtYWlsIjoiampAbm92YW1pbmRueWMuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImZpcmViYXNlIjp7ImlkZW50aXRpZXMiOnsiZW1haWwiOlsiampAbm92YW1pbmRueWMuY29tIl19LCJzaWduX2luX3Byb3ZpZGVyIjoicGFzc3dvcmQifX0.DhWHocoNa20HBlU4ein4veCIWIpw_xqAsHi9cfWlAYfhlbU5wgxUJXGkMGX16tg88x16onEwKVFADIJwvAs-BDF5INIZ60A4H2adDE5OJHNMdDRlwJc6Zc3xDTm5TLRq4dRcu70EVFoT0chzQNS2soMq2p6czg0yiTBCa4H4HpOjYdAsU43pHJ748KpXE6qBYorEdmOmhXB74tHicdl-Z6HwAEFcWm5edljv7vytyCn6P_GKUBSmqPSyCTKCSJ5eQAtQAlU_6irKjetH_na7S92O9f_B8L_YVB8bkkHlKXZ-YxCut_nG51mDBsAe5xBwGOFIypLjywBP_lNU1mBZZQ"

BASE_URL = "https://clarity-backend-282877548076.us-central1.run.app"


def test_endpoint(method, path, auth_required=True, data=None):
    """Test an API endpoint."""
    url = f"{BASE_URL}{path}"
    headers = {}

    if auth_required:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"

    if data:
        headers["Content-Type"] = "application/json"

    print(f"\n{method} {url}")
    print(f"Auth: {'Yes' if auth_required else 'No'}")

    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        else:
            print(f"Unsupported method: {method}")
            return

        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}...")

        if response.status_code >= 400:
            print(f"Full response: {response.text}")

    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run deployment tests."""
    print("=== Testing CLARITY Backend Deployment ===")
    print(f"Base URL: {BASE_URL}")

    # Test health endpoint (no auth)
    test_endpoint("GET", "/health", auth_required=False)

    # Test auth endpoints
    test_endpoint("GET", "/api/v1/auth/me", auth_required=True)

    # Test health data endpoints
    test_endpoint("GET", "/api/v1/health-data", auth_required=True)

    # Test with sample health data upload
    sample_data = {
        "heart_rate": [
            {"timestamp": "2024-01-01T00:00:00Z", "value": 65},
            {"timestamp": "2024-01-01T00:05:00Z", "value": 68},
        ],
        "steps": [
            {"timestamp": "2024-01-01T00:00:00Z", "value": 100},
            {"timestamp": "2024-01-01T00:05:00Z", "value": 250},
        ],
    }
    test_endpoint("POST", "/api/v1/health-data", auth_required=True, data=sample_data)


if __name__ == "__main__":
    main()
