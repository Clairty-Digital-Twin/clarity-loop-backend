#!/usr/bin/env python3
"""Test PAT service integration with secrets manager."""

import os
from pathlib import Path
import sys
from unittest.mock import patch

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from clarity.ml.pat_service import PATModelService
from clarity.security.secrets_manager import get_secrets_manager


def test_pat_service_with_secrets() -> None:
    """Test PAT service uses secrets manager for integrity checks."""
    # Set up test environment
    os.environ["CLARITY_USE_SSM"] = "false"
    os.environ["MODEL_SIGNATURE_KEY"] = "test-signature-key"
    os.environ["EXPECTED_MODEL_CHECKSUMS"] = (
        '{"small": "test-checksum-small", "medium": "test-checksum-medium", "large": "test-checksum-large"}'
    )

    # Clear any cached secrets manager
    import clarity.security.secrets_manager

    clarity.security.secrets_manager._secrets_manager = None

    # Create PAT service
    service = PATModelService(model_size="small")

    # Test 1: Verify it uses secrets manager for checksums

    # Mock the model file to exist
    with patch.object(Path, "exists", return_value=True):
        # Mock the file checksum calculation
        with patch.object(
            PATModelService,
            "_calculate_file_checksum",
            return_value="test-checksum-small",
        ) as mock_calc:
            result = service._verify_model_integrity()
            assert result is True

        # Test with wrong checksum
        with patch.object(
            PATModelService, "_calculate_file_checksum", return_value="wrong-checksum"
        ):
            result = service._verify_model_integrity()
            assert result is False

    # Test 2: Verify signature key is used from secrets

    # Get the secrets manager to verify the key
    manager = get_secrets_manager()
    signature_key = manager.get_model_signature_key()
    assert signature_key == "test-signature-key"

    # Test 3: Test with default values (no env vars)

    # Clear environment
    for key in ["MODEL_SIGNATURE_KEY", "EXPECTED_MODEL_CHECKSUMS"]:
        if key in os.environ:
            del os.environ[key]

    # Clear cached manager
    clarity.security.secrets_manager._secrets_manager = None
    # Also clear the lru_cache on get_secrets_manager
    clarity.security.secrets_manager.get_secrets_manager.cache_clear()

    # Create new service
    service2 = PATModelService(model_size="medium")
    manager2 = get_secrets_manager()

    # Should get defaults
    default_key = manager2.get_model_signature_key()
    default_checksums = manager2.get_model_checksums()

    assert default_key == "pat_model_integrity_key_2025"
    assert "small" in default_checksums
    assert "medium" in default_checksums
    assert "large" in default_checksums


if __name__ == "__main__":
    test_pat_service_with_secrets()
