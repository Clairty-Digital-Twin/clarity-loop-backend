#!/usr/bin/env python3
"""
Test script for the simplified AWS backend
"""
import requests
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = "development-key"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_store_health_data():
    """Test storing health data"""
    print("Testing health data storage...")
    
    headers = {"X-API-Key": API_KEY}
    data = {
        "user_id": "test-user-123",
        "data_type": "heart_rate",
        "value": 72.5,
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": {
            "device": "test-device",
            "activity": "resting"
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/data",
        headers=headers,
        json=data
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    
    return response.json().get("data_id")

def test_get_user_data():
    """Test retrieving user data"""
    print("Testing data retrieval...")
    
    headers = {"X-API-Key": API_KEY}
    response = requests.get(
        f"{BASE_URL}/api/v1/data/test-user-123",
        headers=headers
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_generate_insights():
    """Test generating insights"""
    print("Testing insight generation...")
    
    headers = {"X-API-Key": API_KEY}
    data = {
        "user_id": "test-user-123",
        "query": "What can I do to improve my heart health?",
        "include_recent_data": True
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/insights",
        headers=headers,
        json=data
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_invalid_api_key():
    """Test with invalid API key"""
    print("Testing invalid API key...")
    
    headers = {"X-API-Key": "invalid-key"}
    response = requests.get(
        f"{BASE_URL}/api/v1/data/test-user-123",
        headers=headers
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    print()

if __name__ == "__main__":
    print("=== AWS Simple Backend Test Suite ===\n")
    
    # Run tests
    test_health_check()
    test_store_health_data()
    test_get_user_data()
    test_generate_insights()
    test_invalid_api_key()
    
    print("=== Tests Complete ===")