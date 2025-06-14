#!/usr/bin/env python3
"""Test authentication locally to verify fix before deployment."""

import asyncio
import json
import os
import signal
import subprocess  # noqa: S404 - subprocess needed for server control in test script
import sys
import time

import httpx


async def test_auth() -> bool | None:
    """Test authentication endpoint locally."""
    # Start the server
    print("Starting local server...")
    server_process = subprocess.Popen(
        ["uvicorn", "clarity.main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid,
    )

    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(5)

    try:
        async with httpx.AsyncClient() as client:
            # Test with invalid credentials
            payload = {
                "email": "test@example.com",
                "password": "WrongPassword123!",
                "remember_me": True,
                "device_info": {
                    "device_id": "test-device",
                    "os_version": "test-os",
                    "app_version": "1.0.0",
                },
            }

            print("\nTesting with invalid credentials...")
            response = await client.post(
                "http://localhost:8000/api/v1/auth/login",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")

            if response.status_code == 401:
                print(
                    "\n✅ LOCAL TEST PASSED! Backend correctly returns 401 for invalid credentials"
                )
                print("The fix is working locally. Ready to deploy to production!")
                return True
            print(f"\n❌ LOCAL TEST FAILED! Expected 401, got {response.status_code}")
            return False

    finally:
        # Kill the server
        print("\nStopping server...")
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        server_process.wait()


if __name__ == "__main__":
    result = asyncio.run(test_auth())
    sys.exit(0 if result else 1)
