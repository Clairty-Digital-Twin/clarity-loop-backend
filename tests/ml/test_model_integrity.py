"""Comprehensive tests for model integrity verification.

Tests cover:
- Model validation and verification
- Checkpoint integrity checks
- Model architecture validation
- Security verification
- Error handling and edge cases
"""

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from clarity.ml.model_integrity import (
    ModelIntegrityError,
    ModelIntegrityVerifier,
    ModelSecurityError,
    ModelSignatureError,
    ModelValidationError,
)


@pytest.fixture
def simple_model() -> nn.Module:
    """Create a simple PyTorch model for testing."""
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )


@pytest.fixture
def model_checkpoint(simple_model: nn.Module, tmp_path: Path) -> Path:
    """Create a model checkpoint file."""
    checkpoint_path = tmp_path / "model.pt"
    torch.save(simple_model.state_dict(), checkpoint_path)
    return checkpoint_path


@pytest.fixture
def model_metadata() -> Dict[str, Any]:
    """Sample model metadata."""
    return {
        "model_name": "test_model",
        "version": "1.0.0",
        "architecture": "sequential",
        "input_size": 10,
        "output_size": 1,
        "training_date": "2024-01-01",
        "checksum": "abc123def456",
    }


@pytest.fixture
def verifier() -> ModelIntegrityVerifier:
    """Create model integrity verifier."""
    return ModelIntegrityVerifier()


class TestModelIntegrityVerifier:
    """Test model integrity verifier initialization and basic functionality."""

    @staticmethod
    def test_initialization() -> None:
        """Test verifier initialization."""
        verifier = ModelIntegrityVerifier()
        
        assert verifier.verification_results == {}
        assert verifier.security_checks_enabled is True
        assert verifier.strict_mode is False

    @staticmethod
    def test_initialization_with_options() -> None:
        """Test verifier initialization with custom options."""
        verifier = ModelIntegrityVerifier(
            security_checks_enabled=False,
            strict_mode=True,
        )
        
        assert verifier.security_checks_enabled is False
        assert verifier.strict_mode is True

    @staticmethod
    def test_enable_strict_mode(verifier: ModelIntegrityVerifier) -> None:
        """Test enabling strict mode."""
        verifier.enable_strict_mode()
        
        assert verifier.strict_mode is True

    @staticmethod
    def test_disable_security_checks(verifier: ModelIntegrityVerifier) -> None:
        """Test disabling security checks."""
        verifier.disable_security_checks()
        
        assert verifier.security_checks_enabled is False


class TestModelValidation:
    """Test model validation functionality."""

    @staticmethod
    def test_validate_model_structure_success(verifier: ModelIntegrityVerifier, simple_model: nn.Module) -> None:
        """Test successful model structure validation."""
        expected_structure = {
            "num_layers": 3,
            "layer_types": ["Linear", "ReLU", "Linear"],
            "input_size": 10,
            "output_size": 1,
        }
        
        result = verifier.validate_model_structure(simple_model, expected_structure)
        
        assert result is True

    @staticmethod
    def test_validate_model_structure_mismatch(verifier: ModelIntegrityVerifier, simple_model: nn.Module) -> None:
        """Test model structure validation with mismatch."""
        expected_structure = {
            "num_layers": 5,  # Wrong number of layers
            "layer_types": ["Linear", "ReLU", "Linear"],
            "input_size": 10,
            "output_size": 1,
        }
        
        with pytest.raises(ModelValidationError, match="Model structure validation failed"):
            verifier.validate_model_structure(simple_model, expected_structure)

    @staticmethod
    def test_validate_model_weights_success(verifier: ModelIntegrityVerifier, simple_model: nn.Module) -> None:
        """Test successful model weights validation."""
        # Test that model has reasonable weight ranges
        result = verifier.validate_model_weights(simple_model)
        
        assert result is True

    @staticmethod
    def test_validate_model_weights_invalid_values(verifier: ModelIntegrityVerifier) -> None:
        """Test model weights validation with invalid values."""
        # Create model with NaN weights
        model = nn.Linear(10, 1)
        with torch.no_grad():
            model.weight.fill_(float('nan'))
        
        with pytest.raises(ModelValidationError, match="Model weights contain invalid values"):
            verifier.validate_model_weights(model)

    @staticmethod
    def test_validate_model_weights_extreme_values(verifier: ModelIntegrityVerifier) -> None:
        """Test model weights validation with extreme values."""
        verifier.enable_strict_mode()
        
        # Create model with extremely large weights
        model = nn.Linear(10, 1)
        with torch.no_grad():
            model.weight.fill_(1e10)  # Very large weights
        
        with pytest.raises(ModelValidationError, match="Model weights are outside acceptable range"):
            verifier.validate_model_weights(model)

    @staticmethod
    def test_validate_model_architecture_success(verifier: ModelIntegrityVerifier, simple_model: nn.Module, model_metadata: Dict[str, Any]) -> None:
        """Test successful model architecture validation."""
        result = verifier.validate_model_architecture(simple_model, model_metadata)
        
        assert result is True

    @staticmethod
    def test_validate_model_architecture_mismatch(verifier: ModelIntegrityVerifier, simple_model: nn.Module) -> None:
        """Test model architecture validation with mismatch."""
        wrong_metadata = {
            "architecture": "transformer",  # Wrong architecture
            "input_size": 20,  # Wrong input size
            "output_size": 5,  # Wrong output size
        }
        
        with pytest.raises(ModelValidationError, match="Model architecture validation failed"):
            verifier.validate_model_architecture(simple_model, wrong_metadata)


class TestChecksumVerification:
    """Test checksum verification functionality."""

    @staticmethod
    def test_compute_model_checksum(verifier: ModelIntegrityVerifier, simple_model: nn.Module) -> None:
        """Test model checksum computation."""
        checksum = verifier.compute_model_checksum(simple_model)
        
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex digest length

    @staticmethod
    def test_compute_file_checksum(verifier: ModelIntegrityVerifier, model_checkpoint: Path) -> None:
        """Test file checksum computation."""
        checksum = verifier.compute_file_checksum(model_checkpoint)
        
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex digest length

    @staticmethod
    def test_verify_model_checksum_success(verifier: ModelIntegrityVerifier, simple_model: nn.Module) -> None:
        """Test successful model checksum verification."""
        # Compute expected checksum
        expected_checksum = verifier.compute_model_checksum(simple_model)
        
        result = verifier.verify_model_checksum(simple_model, expected_checksum)
        
        assert result is True

    @staticmethod
    def test_verify_model_checksum_mismatch(verifier: ModelIntegrityVerifier, simple_model: nn.Module) -> None:
        """Test model checksum verification with mismatch."""
        wrong_checksum = "wrong_checksum_value"
        
        with pytest.raises(ModelValidationError, match="Model checksum verification failed"):
            verifier.verify_model_checksum(simple_model, wrong_checksum)

    @staticmethod
    def test_verify_file_checksum_success(verifier: ModelIntegrityVerifier, model_checkpoint: Path) -> None:
        """Test successful file checksum verification."""
        # Compute expected checksum
        expected_checksum = verifier.compute_file_checksum(model_checkpoint)
        
        result = verifier.verify_file_checksum(model_checkpoint, expected_checksum)
        
        assert result is True

    @staticmethod
    def test_verify_file_checksum_mismatch(verifier: ModelIntegrityVerifier, model_checkpoint: Path) -> None:
        """Test file checksum verification with mismatch."""
        wrong_checksum = "wrong_checksum_value"
        
        with pytest.raises(ModelValidationError, match="File checksum verification failed"):
            verifier.verify_file_checksum(model_checkpoint, wrong_checksum)


class TestSecurityChecks:
    """Test security verification functionality."""

    @staticmethod
    def test_verify_model_signature_success(verifier: ModelIntegrityVerifier, simple_model: nn.Module) -> None:
        """Test successful model signature verification."""
        # Mock signature verification
        signature = "mock_signature"
        public_key = "mock_public_key"
        
        with patch.object(verifier, '_verify_digital_signature', return_value=True):
            result = verifier.verify_model_signature(simple_model, signature, public_key)
            
            assert result is True

    @staticmethod
    def test_verify_model_signature_invalid(verifier: ModelIntegrityVerifier, simple_model: nn.Module) -> None:
        """Test model signature verification with invalid signature."""
        signature = "invalid_signature"
        public_key = "mock_public_key"
        
        with patch.object(verifier, '_verify_digital_signature', return_value=False):
            with pytest.raises(ModelSignatureError, match="Model signature verification failed"):
                verifier.verify_model_signature(simple_model, signature, public_key)

    @staticmethod
    def test_check_model_security_success(verifier: ModelIntegrityVerifier, simple_model: nn.Module) -> None:
        """Test successful model security check."""
        result = verifier.check_model_security(simple_model)
        
        assert result is True

    @staticmethod
    def test_check_model_security_malicious_patterns(verifier: ModelIntegrityVerifier) -> None:
        """Test model security check with potential malicious patterns."""
        # Create a model that might trigger security warnings
        class SuspiciousModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(10, 1)
                # Add some attributes that might be flagged
                self.suspicious_attribute = "exec"
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)
        
        suspicious_model = SuspiciousModel()
        
        with pytest.raises(ModelSecurityError, match="Model security check failed"):
            verifier.check_model_security(suspicious_model)

    @staticmethod
    def test_scan_for_malicious_code_success(verifier: ModelIntegrityVerifier, simple_model: nn.Module) -> None:
        """Test successful malicious code scan."""
        result = verifier.scan_for_malicious_code(simple_model)
        
        assert result is True

    @staticmethod
    def test_scan_for_malicious_code_detection(verifier: ModelIntegrityVerifier) -> None:
        """Test malicious code detection."""
        # Create a model with potentially dangerous attributes
        class DangerousModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(10, 1)
                # Add dangerous-looking attributes
                self.eval_func = eval
                self.exec_func = exec
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)
        
        dangerous_model = DangerousModel()
        
        with pytest.raises(ModelSecurityError, match="Potentially malicious code detected"):
            verifier.scan_for_malicious_code(dangerous_model)


class TestComprehensiveVerification:
    """Test comprehensive model verification."""

    @staticmethod
    def test_verify_model_comprehensive_success(verifier: ModelIntegrityVerifier, simple_model: nn.Module, model_metadata: Dict[str, Any]) -> None:
        """Test successful comprehensive model verification."""
        # Update metadata with correct checksum
        model_metadata["checksum"] = verifier.compute_model_checksum(simple_model)
        
        result = verifier.verify_model_comprehensive(simple_model, model_metadata)
        
        assert result is True
        assert "test_model" in verifier.verification_results
        assert verifier.verification_results["test_model"]["status"] == "verified"

    @staticmethod
    def test_verify_model_comprehensive_with_signature(verifier: ModelIntegrityVerifier, simple_model: nn.Module, model_metadata: Dict[str, Any]) -> None:
        """Test comprehensive verification with signature."""
        model_metadata["checksum"] = verifier.compute_model_checksum(simple_model)
        model_metadata["signature"] = "mock_signature"
        model_metadata["public_key"] = "mock_public_key"
        
        with patch.object(verifier, '_verify_digital_signature', return_value=True):
            result = verifier.verify_model_comprehensive(simple_model, model_metadata)
            
            assert result is True

    @staticmethod
    def test_verify_model_comprehensive_security_disabled(verifier: ModelIntegrityVerifier, simple_model: nn.Module, model_metadata: Dict[str, Any]) -> None:
        """Test comprehensive verification with security checks disabled."""
        verifier.disable_security_checks()
        model_metadata["checksum"] = verifier.compute_model_checksum(simple_model)
        
        # Even with suspicious attributes, should pass with security disabled
        class SuspiciousModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(10, 1)
                self.suspicious = "eval"
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)
        
        suspicious_model = SuspiciousModel()
        model_metadata["checksum"] = verifier.compute_model_checksum(suspicious_model)
        
        result = verifier.verify_model_comprehensive(suspicious_model, model_metadata)
        
        assert result is True

    @staticmethod
    def test_verify_checkpoint_file_success(verifier: ModelIntegrityVerifier, model_checkpoint: Path, model_metadata: Dict[str, Any]) -> None:
        """Test successful checkpoint file verification."""
        model_metadata["checksum"] = verifier.compute_file_checksum(model_checkpoint)
        
        result = verifier.verify_checkpoint_file(model_checkpoint, model_metadata)
        
        assert result is True

    @staticmethod
    def test_verify_checkpoint_file_not_found(verifier: ModelIntegrityVerifier, model_metadata: Dict[str, Any]) -> None:
        """Test checkpoint file verification with missing file."""
        nonexistent_path = Path("/nonexistent/model.pt")
        
        with pytest.raises(ModelValidationError, match="Checkpoint file not found"):
            verifier.verify_checkpoint_file(nonexistent_path, model_metadata)


class TestVerificationResults:
    """Test verification results management."""

    @staticmethod
    def test_get_verification_results(verifier: ModelIntegrityVerifier) -> None:
        """Test getting verification results."""
        # Add some mock results
        verifier.verification_results = {
            "model1": {"status": "verified", "timestamp": "2024-01-01"},
            "model2": {"status": "failed", "error": "checksum_mismatch"},
        }
        
        results = verifier.get_verification_results()
        
        assert len(results) == 2
        assert "model1" in results
        assert "model2" in results

    @staticmethod
    def test_get_verification_status(verifier: ModelIntegrityVerifier) -> None:
        """Test getting verification status for specific model."""
        verifier.verification_results["test_model"] = {
            "status": "verified",
            "timestamp": "2024-01-01",
        }
        
        status = verifier.get_verification_status("test_model")
        
        assert status == "verified"

    @staticmethod
    def test_get_verification_status_not_found(verifier: ModelIntegrityVerifier) -> None:
        """Test getting verification status for non-existent model."""
        status = verifier.get_verification_status("nonexistent_model")
        
        assert status is None

    @staticmethod
    def test_clear_verification_results(verifier: ModelIntegrityVerifier) -> None:
        """Test clearing verification results."""
        verifier.verification_results["test"] = {"status": "verified"}
        
        verifier.clear_verification_results()
        
        assert verifier.verification_results == {}


class TestUtilityMethods:
    """Test utility methods."""

    @staticmethod
    def test_verify_digital_signature(verifier: ModelIntegrityVerifier) -> None:
        """Test digital signature verification (mocked)."""
        data = b"test_data"
        signature = "mock_signature"
        public_key = "mock_public_key"
        
        # This is a placeholder implementation
        result = verifier._verify_digital_signature(data, signature, public_key)
        
        # Should return False for mock signature
        assert result is False

    @staticmethod
    def test_extract_model_info(verifier: ModelIntegrityVerifier, simple_model: nn.Module) -> None:
        """Test model information extraction."""
        info = verifier._extract_model_info(simple_model)
        
        assert "architecture" in info
        assert "num_parameters" in info
        assert "layer_info" in info
        assert info["num_parameters"] > 0

    @staticmethod
    def test_validate_metadata(verifier: ModelIntegrityVerifier, model_metadata: Dict[str, Any]) -> None:
        """Test metadata validation."""
        result = verifier._validate_metadata(model_metadata)
        
        assert result is True

    @staticmethod
    def test_validate_metadata_missing_required(verifier: ModelIntegrityVerifier) -> None:
        """Test metadata validation with missing required fields."""
        incomplete_metadata = {
            "model_name": "test_model",
            # Missing other required fields
        }
        
        with pytest.raises(ModelValidationError, match="Missing required metadata"):
            verifier._validate_metadata(incomplete_metadata)


class TestErrorHandling:
    """Test error handling scenarios."""

    @staticmethod
    def test_model_integrity_error_hierarchy() -> None:
        """Test error class hierarchy."""
        # Test that specific errors inherit from base error
        assert issubclass(ModelValidationError, ModelIntegrityError)
        assert issubclass(ModelSecurityError, ModelIntegrityError)
        assert issubclass(ModelSignatureError, ModelIntegrityError)

    @staticmethod
    def test_error_with_details() -> None:
        """Test error creation with details."""
        error = ModelValidationError(
            "Test error",
            model_name="test_model",
            validation_type="checksum",
        )
        
        assert str(error) == "Test error"
        assert error.model_name == "test_model"
        assert error.validation_type == "checksum"

    @staticmethod
    def test_verification_with_corrupt_file(verifier: ModelIntegrityVerifier, tmp_path: Path) -> None:
        """Test verification with corrupted file."""
        # Create a corrupt file
        corrupt_file = tmp_path / "corrupt_model.pt"
        corrupt_file.write_text("not a valid pytorch file")
        
        metadata = {"checksum": "any_checksum"}
        
        with pytest.raises(ModelValidationError):
            verifier.verify_checkpoint_file(corrupt_file, metadata)

    @staticmethod
    def test_verification_with_permission_error(verifier: ModelIntegrityVerifier, tmp_path: Path) -> None:
        """Test verification with permission error."""
        # Create a file and remove read permissions
        restricted_file = tmp_path / "restricted_model.pt"
        restricted_file.write_text("test")
        restricted_file.chmod(0o000)  # No permissions
        
        metadata = {"checksum": "any_checksum"}
        
        try:
            with pytest.raises(ModelValidationError):
                verifier.verify_checkpoint_file(restricted_file, metadata)
        finally:
            # Restore permissions for cleanup
            restricted_file.chmod(0o644)