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


async def test_backend():
    """Test all major endpoints."""
    async with aiohttp.ClientSession() as session:
        print("🚀 Testing CLARITY Backend on AWS\n")

        # Common headers
        headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

        # 1. Test Health Endpoint
        print("1️⃣ Testing Health Check...")
        try:
            async with session.get(f"{BASE_URL}/health", headers=headers) as resp:
                data = await resp.json()
                print(f"✅ Health Check: {resp.status}")
                print(f"   Response: {json.dumps(data, indent=2)}\n")
        except Exception as e:
            print(f"❌ Health Check Failed: {e}\n")

        # 2. Test Auth Health
        print("2️⃣ Testing Auth Service Health...")
        try:
            async with session.get(
                f"{BASE_URL}/api/v1/auth/health", headers=headers
            ) as resp:
                data = await resp.json()
                print(f"✅ Auth Health: {resp.status}")
                print(f"   Response: {json.dumps(data, indent=2)}\n")
        except Exception as e:
            print(f"❌ Auth Health Failed: {e}\n")

        # 3. Test Health Data Service
        print("3️⃣ Testing Health Data Service...")
        try:
            async with session.get(
                f"{BASE_URL}/api/v1/health-data/health", headers=headers
            ) as resp:
                data = await resp.json()
                print(f"✅ Health Data Service: {resp.status}")
                print(f"   Response: {json.dumps(data, indent=2)}\n")
        except Exception as e:
            print(f"❌ Health Data Service Failed: {e}\n")

        # 4. Test User Registration (Mock)
        print("4️⃣ Testing User Registration...")
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
                print(f"✅ Registration: {resp.status}")
                print(f"   Response: {json.dumps(data, indent=2)}\n")
        except Exception as e:
            print(f"❌ Registration Failed: {e}\n")

        # 5. Test Login
        print("5️⃣ Testing User Login...")
        login_data = {"email": test_user["email"], "password": test_user["password"]}
        auth_token = None
        try:
            async with session.post(
                f"{BASE_URL}/api/v1/auth/login", headers=headers, json=login_data
            ) as resp:
                data = await resp.json()
                print(f"✅ Login: {resp.status}")
                if "access_token" in data:
                    auth_token = data["access_token"]
                    print(f"   Token received: {auth_token[:20]}...")
                print(f"   Response: {json.dumps(data, indent=2)}\n")
        except Exception as e:
            print(f"❌ Login Failed: {e}\n")
            auth_token = "mock-jwt-token"  # Fallback for testing

        # Update headers with auth token
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        # 6. Test Health Data Upload
        print("6️⃣ Testing Health Data Upload...")
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
                print(f"✅ Health Data Upload: {resp.status}")
                print(f"   Response: {json.dumps(data, indent=2)}\n")
        except Exception as e:
            print(f"❌ Health Data Upload Failed: {e}\n")

        # 7. Test Health Data Query
        print("7️⃣ Testing Health Data Query...")
        try:
            params = {"limit": 10, "data_type": "heart_rate"}
            async with session.get(
                f"{BASE_URL}/api/v1/health-data/", headers=headers, params=params
            ) as resp:
                data = await resp.json()
                print(f"✅ Health Data Query: {resp.status}")
                if "data" in data:
                    print(f"   Records returned: {len(data['data'])}")
                print(f"   Response preview: {json.dumps(data, indent=2)[:200]}...\n")
        except Exception as e:
            print(f"❌ Health Data Query Failed: {e}\n")

        # 8. Test Insights Generation
        print("8️⃣ Testing AI Insights Generation...")
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
                print(f"✅ Insights Generation: {resp.status}")
                if "data" in data and "narrative" in data["data"]:
                    print(f"   Narrative preview: {data['data']['narrative'][:100]}...")
                print(f"   Response: {json.dumps(data, indent=2)[:300]}...\n")
        except Exception as e:
            print(f"❌ Insights Generation Failed: {e}\n")

        # 9. Test Current User Profile
        print("9️⃣ Testing User Profile...")
        try:
            async with session.get(
                f"{BASE_URL}/api/v1/auth/me", headers=headers
            ) as resp:
                data = await resp.json()
                print(f"✅ User Profile: {resp.status}")
                print(f"   Response: {json.dumps(data, indent=2)}\n")
        except Exception as e:
            print(f"❌ User Profile Failed: {e}\n")

        # 10. Test API Documentation
        print("🔟 Testing API Documentation...")
        try:
            async with session.get(f"{BASE_URL}/docs", headers=headers) as resp:
                print(f"✅ Swagger UI: {resp.status}")
        except Exception as e:
            print(f"❌ Swagger UI Failed: {e}")

        try:
            async with session.get(f"{BASE_URL}/openapi.json", headers=headers) as resp:
                data = await resp.json()
                print(f"✅ OpenAPI Schema: {resp.status}")
                print(f"   API Title: {data.get('info', {}).get('title')}")
                print(f"   Version: {data.get('info', {}).get('version')}\n")
        except Exception as e:
            print(f"❌ OpenAPI Schema Failed: {e}\n")

        print("✨ Test Suite Complete!")


if __name__ == "__main__":
    print("=" * 60)
    print("CLARITY AWS Backend Test Suite")
    print(f"Target: {BASE_URL}")
    print(f"API Key: {'*' * 20}{API_KEY[-10:]}")
    print("=" * 60 + "\n")

    asyncio.run(test_backend())
