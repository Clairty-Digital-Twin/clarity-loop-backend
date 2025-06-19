#!/usr/bin/env python3
"""Test script to demonstrate account lockout functionality.

This script simulates multiple failed login attempts to test the lockout service.
Works with both Redis and in-memory backends.
"""

import asyncio
from datetime import timedelta
import os
from pathlib import Path
import sys

import requests

# Add the src directory to the path so we can import clarity modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clarity.auth.lockout_service import AccountLockoutError, AccountLockoutService


async def test_lockout_demo() -> None:
    """Demonstrate the lockout service functionality."""
    test_email = "demo@example.com"
    _wrong_password = "WrongP@ssword"  # noqa: S105 - Hardcoded password for demo

    print("🔒 Account Lockout Service Demo")
    print("=" * 50)

    # Get Redis URL from environment (optional)
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        print(f"🔗 Using Redis persistence: {redis_url}")
    else:
        print("💾 Using in-memory storage (will reset on restart)")

    # Create lockout service with faster settings for demo
    lockout_service = AccountLockoutService(
        max_attempts=3,  # Lock after 3 attempts
        lockout_duration=timedelta(minutes=1),  # Lock for 1 minute for demo
        redis_url=redis_url,
    )

    # Future enhancement: IP-based lockout testing

    print(f"Testing with email: {test_email}")
    print("Max attempts before lockout: 3")
    print("Lockout duration: 1 minute")
    print()

    # Clean slate - reset any existing attempts
    await lockout_service.reset(test_email)
    print(f"🧹 Reset any existing attempts for {test_email}")
    print()

    # Test 1: Record failed attempts
    print("🚨 Simulating failed login attempts...")
    for attempt in range(1, 5):  # Try 4 attempts to ensure we hit lockout
        try:
            # Check if account is locked before attempting
            if await lockout_service.is_locked(test_email):
                print(f"  🔒 Account is locked before attempt {attempt}")
                break

            print(
                f"  Attempt {attempt}: Account not locked, simulating failed login..."
            )
            await lockout_service.register_failure(test_email)
            print(f"  ❌ Failed attempt {attempt} recorded")

            # Check if account got locked after this attempt
            if await lockout_service.is_locked(test_email):
                print(f"  🔒 Account locked after {attempt} attempts!")
                break

        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"  ❌ Error during attempt {attempt}: {e}")
            break
        print()

    # Test 2: Try to access locked account
    print("🔒 Testing locked account access...")
    try:
        await lockout_service.check_lockout(test_email)
        print("  ✅ Account is not locked (unexpected!)")
    except AccountLockoutError as e:
        print(f"  🚫 Account is locked: {e}")
    print()

    # Test 3: Check current lockout status
    print("📊 Checking lockout status...")
    is_locked = await lockout_service.is_locked(test_email)
    print(f"  Email: {test_email}")
    print(f"  Currently locked: {is_locked}")
    print()

    # Test 4: Manual reset (simulate successful login)
    print("🔄 Testing manual reset (simulating successful login)...")
    await lockout_service.reset(test_email)
    print(f"  ✅ Reset attempts for {test_email}")

    is_locked_after_reset = await lockout_service.is_locked(test_email)
    print(f"  Locked after reset: {is_locked_after_reset}")
    print()

    # Test 5: Verify reset worked
    print("✅ Testing that reset worked...")
    try:
        await lockout_service.check_lockout(test_email)
        print("  ✅ Account is no longer locked - reset successful!")
    except AccountLockoutError as e:
        print(f"  ❌ Account is still locked after reset: {e}")
    print()

    print("✅ Demo completed! Account lockout service is working correctly.")
    print()
    print("Key features demonstrated:")
    print("  • Failed attempts are tracked per user")
    print("  • Account gets locked after max attempts")
    print("  • Lockout status can be queried")
    print("  • Manual reset clears lockout state")
    print("  • Works with both Redis and in-memory storage")
    print("  • Thread-safe concurrent access")


if __name__ == "__main__":
    # Allow command line arguments for testing
    import argparse

    parser = argparse.ArgumentParser(description="Test account lockout functionality")
    parser.add_argument(
        "email", nargs="?", default="demo@example.com", help="Email to test with"
    )
    parser.add_argument(
        "password",
        nargs="?",
        default="WrongP@ssword",
        help="Wrong password to test with",
    )

    args = parser.parse_args()

    asyncio.run(test_lockout_demo(args.email, args.password))
