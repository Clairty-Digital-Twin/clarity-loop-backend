#!/usr/bin/env python3
"""Test script for the simplified AWS backend."""
from datetime import datetime
import json

import requests

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = "development-key"


def test_health_check() -> None:
    """Test the health check endpoint."""
    response = requests.get(f"{BASE_URL}/health")


def test_store_health_data():
    """Test storing health data."""
    headers = {"X-API-Key": API_KEY}
    data = {
        "user_id": "test-user-123",
        "data_type": "heart_rate",
        "value": 72.5,
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": {"device": "test-device", "activity": "resting"},
    }

    response = requests.post(f"{BASE_URL}/api/v1/data", headers=headers, json=data)

    return response.json().get("data_id")


def test_get_user_data() -> None:
    """Test retrieving user data."""
    headers = {"X-API-Key": API_KEY}
    response = requests.get(f"{BASE_URL}/api/v1/data/test-user-123", headers=headers)


def test_generate_insights() -> None:
    """Test generating insights."""
    headers = {"X-API-Key": API_KEY}
    data = {
        "user_id": "test-user-123",
        "query": "What can I do to improve my heart health?",
        "include_recent_data": True,
    }

    response = requests.post(f"{BASE_URL}/api/v1/insights", headers=headers, json=data)


def test_invalid_api_key() -> None:
    """Test with invalid API key."""
    headers = {"X-API-Key": "invalid-key"}
    response = requests.get(f"{BASE_URL}/api/v1/data/test-user-123", headers=headers)


if __name__ == "__main__":

    # Run tests
    test_health_check()
    test_store_health_data()
    test_get_user_data()
    test_generate_insights()
    test_invalid_api_key()
