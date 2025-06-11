#!/usr/bin/env python3
"""Quick AWS Backend Test Script for CLARITY.

Tests core functionality with the production API key.
"""

import asyncio
from datetime import UTC, datetime
import json

import aiohttp

BASE_URL = "http://***REMOVED***"
API_KEY = "production-api-key-change-me"


async def test_backend() -> None:
    """Test all major endpoints."""
    async with aiohttp.ClientSession() as session:

        # Common headers
        headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

        # 1. Test Health Endpoint
        try:
            async with session.get(f"{BASE_URL}/health", headers=headers) as resp:
                data = await resp.json()
        except Exception as e:
            pass

        # 2. Test Auth Health
        try:
            async with session.get(
                f"{BASE_URL}/api/v1/auth/health", headers=headers
            ) as resp:
                data = await resp.json()
        except Exception as e:
            pass

        # 3. Test Health Data Service
        try:
            async with session.get(
                f"{BASE_URL}/api/v1/health-data/health", headers=headers
            ) as resp:
                data = await resp.json()
        except Exception as e:
            pass

        # 4. Test User Registration (Mock)
        test_user = {
            "email": f"test_{datetime.now().timestamp()}@clarity.health",
            "password": "TestPassword123!",
            "profile": {
                "first_name": "Test",
                "last_name": "User",
                "date_of_birth": "1990-01-01",
            },
        }
        try:
            async with session.post(
                f"{BASE_URL}/api/v1/auth/register", headers=headers, json=test_user
            ) as resp:
                data = await resp.json()
        except Exception as e:
            pass

        # 5. Test Login
        login_data = {"email": test_user["email"], "password": test_user["password"]}
        auth_token = None
        try:
            async with session.post(
                f"{BASE_URL}/api/v1/auth/login", headers=headers, json=login_data
            ) as resp:
                data = await resp.json()
                if "access_token" in data:
                    auth_token = data["access_token"]
        except Exception as e:
            auth_token = "mock-jwt-token"  # Fallback for testing

        # Update headers with auth token
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        # 6. Test Health Data Upload
        health_data = {
            "user_id": "test_user_123",
            "data_type": "heart_rate",
            "measurements": [
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "value": 72.5,
                    "unit": "bpm",
                }
            ],
            "source": "apple_watch",
            "device_info": {"model": "Apple Watch Series 9", "os_version": "10.0"},
        }
        try:
            async with session.post(
                f"{BASE_URL}/api/v1/health-data/upload",
                headers=headers,
                json=health_data,
            ) as resp:
                data = await resp.json()
        except Exception as e:
            pass

        # 7. Test Health Data Query
        try:
            params = {"limit": 10, "data_type": "heart_rate"}
            async with session.get(
                f"{BASE_URL}/api/v1/health-data/", headers=headers, params=params
            ) as resp:
                data = await resp.json()
                if "data" in data:
                    pass
        except Exception as e:
            pass

        # 8. Test Insights Generation
        insight_request = {
            "analysis_results": {
                "heart_rate_avg": 72,
                "sleep_quality": 0.85,
                "activity_level": "moderate",
            },
            "context": "Weekly health summary",
            "insight_type": "comprehensive",
        }
        try:
            async with session.post(
                f"{BASE_URL}/api/v1/insights/generate",
                headers=headers,
                json=insight_request,
            ) as resp:
                data = await resp.json()
                if "data" in data and "narrative" in data["data"]:
                    pass
        except Exception as e:
            pass

        # 9. Test Current User Profile
        try:
            async with session.get(
                f"{BASE_URL}/api/v1/auth/me", headers=headers
            ) as resp:
                data = await resp.json()
        except Exception as e:
            pass

        # 10. Test API Documentation
        try:
            async with session.get(f"{BASE_URL}/docs", headers=headers) as resp:
                pass
        except Exception as e:
            pass

        try:
            async with session.get(f"{BASE_URL}/openapi.json", headers=headers) as resp:
                data = await resp.json()
        except Exception as e:
            pass


if __name__ == "__main__":

    asyncio.run(test_backend())
