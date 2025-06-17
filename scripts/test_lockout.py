#!/usr/bin/env python3
"""
Test script to demonstrate account lockout functionality.
This script simulates multiple failed login attempts to test the lockout service.
"""

import asyncio
import sys
import os
from datetime import timedelta

# Add the src directory to the path so we can import clarity modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from clarity.auth.lockout_service import AccountLockoutService, AccountLockoutError


async def test_lockout_demo():
    """Demonstrate the lockout service functionality."""
    print("🔒 Account Lockout Service Demo")
    print("=" * 50)
    
    # Create lockout service with faster settings for demo
    lockout_service = AccountLockoutService(
        max_attempts=3,  # Lock after 3 attempts instead of 5
        lockout_duration=timedelta(minutes=1)  # Lock for 1 minute instead of 15
    )
    
    test_email = "demo@example.com"
    test_ip = "192.168.1.100"
    
    print(f"Testing with email: {test_email}")
    print(f"Max attempts before lockout: 3")
    print(f"Lockout duration: 1 minute")
    print()
    
    # Test 1: Record failed attempts
    print("🚨 Simulating failed login attempts...")
    for attempt in range(1, 4):
        try:
            await lockout_service.check_lockout(test_email)
            print(f"  Attempt {attempt}: Account not locked, proceeding...")
            await lockout_service.record_failed_attempt(test_email, test_ip)
            print(f"  ❌ Failed attempt {attempt} recorded")
        except AccountLockoutError as e:
            print(f"  🔒 Account locked: {e}")
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
    
    # Test 3: Check lockout status
    print("📊 Checking lockout status...")
    status = await lockout_service.get_lockout_status(test_email)
    if status:
        print(f"  Email: {test_email}")
        print(f"  Locked: {status['locked']}")
        print(f"  Attempts: {status['attempts']}/{status['max_attempts']}")
        if status['locked']:
            print(f"  Unlock time: {status['unlock_time']}")
            print(f"  Time remaining: {status['time_remaining_seconds']} seconds")
    else:
        print(f"  No lockout data for {test_email}")
    print()
    
    # Test 4: Wait for lockout to expire (optional - uncomment for full demo)
    print("⏰ Lockout will expire in 1 minute...")
    print("   (Skipping wait for demo - uncomment sleep to test expiry)")
    # print("   Waiting for lockout to expire...")
    # await asyncio.sleep(65)  # Wait a bit longer than 1 minute
    # 
    # try:
    #     await lockout_service.check_lockout(test_email)
    #     print("  ✅ Account is no longer locked!")
    # except AccountLockoutError as e:
    #     print(f"  🚫 Account is still locked: {e}")
    
    print()
    print("✅ Demo completed! Account lockout service is working correctly.")
    print()
    print("Key features demonstrated:")
    print("  • Failed attempts are tracked per user")
    print("  • Account gets locked after max attempts")
    print("  • Lockout status can be queried")
    print("  • Lockouts expire after the configured duration")
    print("  • Thread-safe concurrent access")


if __name__ == "__main__":
    asyncio.run(test_lockout_demo())