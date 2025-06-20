#!/usr/bin/env python3
"""Test script to verify security headers are working."""

import subprocess  # noqa: S404 - Required for testing
import sys
import time

import requests

# Constants
MAX_DISPLAY_LENGTH = 60

# Start the server in the background
print("Starting server...")
server_process = subprocess.Popen(  # noqa: S603 - Testing script
    [
        sys.executable,
        "-m",
        "uvicorn",
        "clarity.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
    ],
    cwd="src",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Give the server time to start
time.sleep(3)

try:
    # Test the health endpoint
    response = requests.get("http://localhost:8001/health", timeout=10)
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.json()}")
    print("\nSecurity Headers:")

    headers_to_check = [
        "Strict-Transport-Security",
        "X-Content-Type-Options",
        "X-Frame-Options",
        "Content-Security-Policy",
        "X-XSS-Protection",
        "Referrer-Policy",
        "Cache-Control",
        "Permissions-Policy",
    ]

    found_count = 0
    for header in headers_to_check:
        if header in response.headers:
            value = response.headers[header]
            display_value = (
                f"{value[:MAX_DISPLAY_LENGTH]}..."
                if len(value) > MAX_DISPLAY_LENGTH
                else value
            )
            print(f"✅ {header}: {display_value}")
            found_count += 1
        else:
            print(f"❌ {header}: NOT FOUND")

    print(f"\nTotal headers found: {found_count}/{len(headers_to_check)}")

    # Test the root endpoint
    response = requests.get("http://localhost:8001/", timeout=10)
    print(f"\nRoot endpoint status: {response.status_code}")
    print(
        f'Has security headers: {"Yes" if "X-Content-Type-Options" in response.headers else "No"}'
    )

except (requests.RequestException, ValueError, KeyError) as e:
    print(f"Error: {e}")

finally:
    # Stop the server
    print("\nStopping server...")
    server_process.terminate()
    server_process.wait()
    print("Test complete!")
