"""Integration tests for PAT service with secrets manager."""

import os
from unittest.mock import patch, mock_open

import pytest

from clarity.ml.pat_service import PATModelService
from clarity.security.secrets_manager import get_secrets_manager


class TestPATServiceSecretsIntegration:
    """Test PAT service integration with secrets manager."""

    def test_pat_service_uses_secrets_manager_for_checksums(self):
        """Test that PAT service retrieves checksums from secrets manager."""
        # Set up test checksums via environment
        test_checksums = {
            "small": "test-checksum-small",
            "medium": "test-checksum-medium",
            "large": "test-checksum-large",
        }
        
        with patch.dict(
            os.environ,
            {
                "CLARITY_USE_SSM": "false",
                "EXPECTED_MODEL_CHECKSUMS": '{"small": "test-checksum-small", "medium": "test-checksum-medium", "large": "test-checksum-large"}',
            },
        ):
            # Clear any cached secrets manager
            import clarity.security.secrets_manager
            clarity.security.secrets_manager._secrets_manager = None
            
            # Create PAT service
            service = PATModelService(model_size="small")
            
            # Mock the file checksum calculation to return our test checksum
            with patch.object(
                PATModelService,
                "_calculate_file_checksum",
                return_value="test-checksum-small",
            ):
                # Verify integrity should pass with matching checksum
                assert service._verify_model_integrity() is True
            
            # Test with non-matching checksum
            with patch.object(
                PATModelService,
                "_calculate_file_checksum",
                return_value="different-checksum",
            ):
                # Verify integrity should fail with non-matching checksum
                assert service._verify_model_integrity() is False

    def test_pat_service_uses_secrets_manager_for_signature_key(self):
        """Test that PAT service retrieves signature key from secrets manager."""
        test_signature_key = "test-signature-key-123"
        
        with patch.dict(
            os.environ,
            {
                "CLARITY_USE_SSM": "false",
                "MODEL_SIGNATURE_KEY": test_signature_key,
            },
        ):
            # Clear any cached secrets manager
            import clarity.security.secrets_manager
            clarity.security.secrets_manager._secrets_manager = None
            
            # Create PAT service
            service = PATModelService(model_size="medium")
            
            # Mock the file reading to avoid actual file I/O
            from pathlib import Path
            test_path = Path("/test/model.h5")
            
            with patch("pathlib.Path.open", mock_open(read_data=b"test model data")):
                with patch("pathlib.Path.exists", return_value=True):
                    # Calculate checksum should use the signature key from secrets
                    with patch("hashlib.sha256") as mock_sha256:
                        mock_sha256.return_value.hexdigest.return_value = "test_digest"
                        with patch("hmac.new") as mock_hmac:
                            mock_hmac.return_value.hexdigest.return_value = "test_signature"
                            # Call the static method directly
                            result = PATModelService._calculate_file_checksum(test_path)
                            
                            # Verify HMAC was called (it gets called multiple times by AWS SDK)
                            assert mock_hmac.called
                            # Find the call that used our test signature key
                            found_our_call = False
                            for call in mock_hmac.call_args_list:
                                if len(call[0]) > 0 and call[0][0] == b"test-signature-key-123":
                                    found_our_call = True
                                    break
                            assert found_our_call, "HMAC was not called with our test signature key"
                            assert result == "test_signature"

    def test_pat_service_fallback_when_secrets_unavailable(self):
        """Test PAT service falls back gracefully when secrets are unavailable."""
        # Clear environment to simulate no secrets available
        env_vars_to_clear = [
            "CLARITY_USE_SSM",
            "MODEL_SIGNATURE_KEY",
            "EXPECTED_MODEL_CHECKSUMS",
            "CLARITY_MODEL_SIGNATURE_KEY",
            "CLARITY_EXPECTED_MODEL_CHECKSUMS",
        ]
        
        # Use patch.dict to ensure clean environment isolation
        clean_env = os.environ.copy()
        for var in env_vars_to_clear:
            if var in clean_env:
                del clean_env[var]
        
        with patch.dict(os.environ, clean_env, clear=True):
            # Clear any cached secrets manager
            import clarity.security.secrets_manager
            clarity.security.secrets_manager._secrets_manager = None
            
            # Create PAT service - should use defaults
            service = PATModelService(model_size="large")
            
            # Get secrets manager and verify defaults are used
            manager = get_secrets_manager()
            signature_key = manager.get_model_signature_key()
            checksums = manager.get_model_checksums()
            
            # Should get default values
            assert signature_key == "pat_model_integrity_key_2025"
            assert "small" in checksums
            assert "medium" in checksums
            assert "large" in checksums

    @pytest.mark.parametrize("model_size", ["small", "medium", "large"])
    def test_pat_service_integrity_check_with_real_checksums(self, model_size):
        """Test PAT service integrity check with real default checksums."""
        # Use the real default checksums
        real_checksums = {
            "small": "4b30d57febbbc8ef221e4b196bf6957e7c7f366f6b836fe800a43f69d24694ad",
            "medium": "6175021ca1a43f3c834bdaa644c45f27817cf985d8ffd186fab9b5de2c4ca661",
            "large": "c93b723f297f0d9d2ad982320b75e9212882c8f38aa40df1b600e9b2b8aa1973",
        }
        
        with patch.dict(
            os.environ,
            {
                "CLARITY_USE_SSM": "false",
                "MODEL_SIGNATURE_KEY": "pat_model_integrity_key_2025",
            },
        ):
            # Clear any cached secrets manager
            import clarity.security.secrets_manager
            clarity.security.secrets_manager._secrets_manager = None
            
            # Create PAT service
            service = PATModelService(model_size=model_size)
            
            # Mock file existence and checksum to return the expected checksum
            with patch("pathlib.Path.exists", return_value=True):
                with patch.object(
                    PATModelService,
                    "_calculate_file_checksum",
                    return_value=real_checksums[model_size],
                ):
                    # Verify integrity should pass
                    assert service._verify_model_integrity() is True