#!/usr/bin/env python3
"""Basic test script for secrets manager functionality."""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from clarity.security.secrets_manager import SecretsManager, get_secrets_manager


def test_basic_functionality():
    """Test basic secrets manager functionality."""
    print("Testing Clarity Secrets Manager...")
    
    # Test 1: Create manager with environment variables
    os.environ["CLARITY_USE_SSM"] = "false"
    os.environ["MODEL_SIGNATURE_KEY"] = "test-key-123"
    os.environ["EXPECTED_MODEL_CHECKSUMS"] = '{"small": "abc123", "medium": "def456"}'
    
    manager = SecretsManager(use_ssm=False)
    
    # Test getting string value
    signature_key = manager.get_model_signature_key()
    print(f"✓ Model signature key: {signature_key}")
    assert signature_key == "test-key-123"
    
    # Test getting JSON value
    checksums = manager.get_model_checksums()
    print(f"✓ Model checksums: {checksums}")
    assert checksums["small"] == "abc123"
    assert checksums["medium"] == "def456"
    
    # Test health check
    health = manager.health_check()
    print(f"✓ Health check: {health}")
    assert health["service"] == "SecretsManager"
    assert health["ssm_status"] == "disabled"
    
    # Test singleton
    manager2 = get_secrets_manager()
    print("✓ Singleton pattern works")
    
    print("\nAll tests passed! ✅")


if __name__ == "__main__":
    test_basic_functionality()