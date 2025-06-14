#!/usr/bin/env python3
import httpx
import asyncio
import json

async def test_auth():
    """Test authentication endpoint."""
    async with httpx.AsyncClient() as client:
        # Test with invalid credentials
        payload = {
            "email": "test@example.com",
            "password": "WrongPassword123!",
            "remember_me": True,
            "device_info": {
                "device_id": "test-device",
                "os_version": "test-os",
                "app_version": "1.0.0"
            }
        }
        
        print("Testing with invalid credentials...")
        response = await client.post(
            "http://localhost:8000/api/v1/auth/login",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 401:
            print("✅ Test passed! Backend correctly returns 401 for invalid credentials")
        else:
            print(f"❌ Test failed! Expected 401, got {response.status_code}")

if __name__ == "__main__":
    asyncio.run(test_auth())