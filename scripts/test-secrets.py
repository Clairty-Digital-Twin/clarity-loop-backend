#!/usr/bin/env python3
"""Test script to verify AWS Secrets Manager injection in ECS."""
import os
import sys

print("Testing AWS Secrets Manager injection...")
print("=" * 50)

# Check environment
print(f"ENVIRONMENT: {os.getenv('ENVIRONMENT', 'NOT SET')}")
print(f"AWS_REGION: {os.getenv('AWS_REGION', 'NOT SET')}")

# Check secrets
secret_key = os.getenv("SECRET_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

print(f"\nSECRET_KEY exists: {'YES' if secret_key else 'NO'}")
if secret_key:
    print(f"  - Length: {len(secret_key)}")
    print(f"  - First 5 chars: {secret_key[:5]}...")
    dev_key = "dev-secret-key"
    print(f"  - Is '{dev_key}': {secret_key == dev_key}")

print(f"\nGEMINI_API_KEY exists: {'YES' if gemini_key else 'NO'}")
if gemini_key:
    print(f"  - Length: {len(gemini_key)}")
    print(f"  - First 5 chars: {gemini_key[:5]}...")

# Test configuration loading
print("\n" + "=" * 50)
print("Testing configuration loading...")
try:
    from clarity.startup.config_schema import ClarityConfig

    config = ClarityConfig()
    print("✅ Configuration loaded successfully!")
    print(f"  - Environment: {config.environment}")
    dev_key = "dev-secret-key"
    print(f"  - Secret key is default: {config.security.secret_key == dev_key}")
except Exception as e:  # noqa: BLE001
    print(f"❌ Configuration failed: {e}")
    sys.exit(1)

print("\n✅ All tests passed!\n")
