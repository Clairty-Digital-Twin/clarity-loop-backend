"""Integration tests for model signing and verification."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the signing module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from sign_model import ModelSigner, ModelSigningError, get_secret_key


class TestModelSigner:
    """Test cases for ModelSigner class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
            
    @pytest.fixture
    def test_model_file(self, temp_dir):
        """Create a test model file."""
        model_path = temp_dir / "test_model.pth"
        model_path.write_bytes(b"fake model data for testing")
        return model_path
        
    @pytest.fixture
    def signer(self):
        """Create a ModelSigner instance with test key."""
        return ModelSigner("test-secret-key-12345")
        
    def test_sign_model_creates_signature_file(self, signer, test_model_file):
        """Test that signing creates a .sig file."""
        # Sign the model
        metadata = signer.sign_model(test_model_file)
        
        # Check signature file exists
        sig_path = test_model_file.with_suffix(".pth.sig")
        assert sig_path.exists()
        
        # Verify metadata
        assert metadata["file"] == "test_model.pth"
        assert metadata["algorithm"] == "sha256"
        assert "signature" in metadata
        assert "signed_at" in metadata
        assert metadata["file_size"] == 27  # len(b"fake model data for testing")
        
    def test_verify_model_with_valid_signature(self, signer, test_model_file):
        """Test verification of a properly signed model."""
        # Sign the model
        signer.sign_model(test_model_file)
        
        # Verify it
        result, error = signer.verify_model(test_model_file)
        assert result is True
        assert error is None
        
    def test_verify_model_with_modified_file(self, signer, test_model_file):
        """Test that verification fails when model is modified."""
        # Sign the model
        signer.sign_model(test_model_file)
        
        # Modify the model file
        test_model_file.write_bytes(b"modified model data")
        
        # Verify should fail
        result, error = signer.verify_model(test_model_file)
        assert result is False
        assert "Signature mismatch" in error
        
    def test_verify_model_with_wrong_key(self, test_model_file):
        """Test that verification fails with wrong key."""
        # Sign with one key
        signer1 = ModelSigner("key1")
        signer1.sign_model(test_model_file)
        
        # Verify with different key
        signer2 = ModelSigner("key2")
        result, error = signer2.verify_model(test_model_file)
        assert result is False
        assert "Signature mismatch" in error
        
    def test_verify_model_without_signature(self, signer, test_model_file):
        """Test verification when signature file is missing."""
        result, error = signer.verify_model(test_model_file)
        assert result is False
        assert "Signature file not found" in error
        
    def test_sign_nonexistent_model(self, signer, temp_dir):
        """Test signing a non-existent file raises error."""
        fake_path = temp_dir / "nonexistent.pth"
        with pytest.raises(ModelSigningError, match="Model file not found"):
            signer.sign_model(fake_path)
            
    def test_sign_directory(self, signer, temp_dir):
        """Test signing multiple models in a directory."""
        # Create test models
        models = []
        for i in range(3):
            model_path = temp_dir / f"model_{i}.pth"
            model_path.write_bytes(f"model {i} data".encode())
            models.append(model_path)
            
        # Also create a non-model file
        other_file = temp_dir / "readme.txt"
        other_file.write_text("not a model")
        
        # Sign directory
        signatures = signer.sign_directory(temp_dir)
        
        # Check results
        assert len(signatures) == 3
        for i in range(3):
            assert f"model_{i}.pth" in signatures
            
        # Check manifest file
        manifest_path = temp_dir / "model_signatures.json"
        assert manifest_path.exists()
        
        with manifest_path.open() as f:
            manifest = json.load(f)
            assert manifest["total_files"] == 3
            assert manifest["algorithm"] == "sha256"
            
    def test_verify_directory(self, signer, temp_dir):
        """Test verifying multiple models in a directory."""
        # Create and sign test models
        for i in range(3):
            model_path = temp_dir / f"model_{i}.pth"
            model_path.write_bytes(f"model {i} data".encode())
            signer.sign_model(model_path)
            
        # Verify all should pass
        passed, failed, failed_files = signer.verify_directory(temp_dir)
        assert passed == 3
        assert failed == 0
        assert failed_files == []
        
        # Modify one model
        (temp_dir / "model_1.pth").write_bytes(b"tampered data")
        
        # Verify should show one failure
        passed, failed, failed_files = signer.verify_directory(temp_dir)
        assert passed == 2
        assert failed == 1
        assert "model_1.pth" in failed_files
        
    def test_file_size_verification(self, signer, test_model_file):
        """Test that file size changes are detected."""
        # Sign the model
        signer.sign_model(test_model_file)
        
        # Modify signature file to have wrong size
        sig_path = test_model_file.with_suffix(".pth.sig")
        with sig_path.open() as f:
            metadata = json.load(f)
            
        metadata["file_size"] = 9999  # Wrong size
        
        with sig_path.open("w") as f:
            json.dump(metadata, f)
            
        # Verification should fail
        result, error = signer.verify_model(test_model_file)
        assert result is False
        assert "File size mismatch" in error
        
    def test_recursive_directory_signing(self, signer, temp_dir):
        """Test recursive signing of nested directories."""
        # Create nested structure
        subdir = temp_dir / "submodels"
        subdir.mkdir()
        
        model1 = temp_dir / "model1.pth"
        model1.write_bytes(b"model 1")
        
        model2 = subdir / "model2.onnx"
        model2.write_bytes(b"model 2")
        
        # Sign recursively
        signatures = signer.sign_directory(temp_dir, recursive=True)
        assert len(signatures) == 2
        assert "model1.pth" in signatures
        assert str(Path("submodels/model2.onnx")) in signatures
        
        # Sign non-recursively
        signatures = signer.sign_directory(temp_dir, recursive=False)
        assert len(signatures) == 1
        assert "model1.pth" in signatures


class TestSecretKeyHandling:
    """Test cases for secret key management."""
    
    def test_get_secret_key_from_env(self):
        """Test getting key from environment variable."""
        with patch.dict(os.environ, {"MODEL_SIGNING_KEY": "env-secret-key"}):
            key = get_secret_key()
            assert key == "env-secret-key"
            
    def test_get_secret_key_ci_mode_without_env(self):
        """Test that CI mode fails without environment variable."""
        with patch.dict(os.environ, {"CI": "true"}, clear=True):
            with pytest.raises(ValueError, match="MODEL_SIGNING_KEY environment variable not set"):
                get_secret_key()
                
    def test_empty_secret_key_raises_error(self):
        """Test that empty secret key is rejected."""
        with pytest.raises(ValueError, match="Secret key cannot be empty"):
            ModelSigner("")


class TestCLIIntegration:
    """Test command-line interface integration."""
    
    @pytest.fixture
    def cli_env(self, temp_dir):
        """Set up environment for CLI tests."""
        # Create test model
        model_path = temp_dir / "test_model.pth"
        model_path.write_bytes(b"test model data")
        
        # Set secret key
        env = os.environ.copy()
        env["MODEL_SIGNING_KEY"] = "cli-test-key"
        
        return {
            "model_path": model_path,
            "temp_dir": temp_dir,
            "env": env,
        }
        
    def test_cli_sign_command(self, cli_env):
        """Test CLI sign command."""
        import subprocess
        
        result = subprocess.run(
            [
                sys.executable,
                "scripts/sign_model.py",
                "sign",
                "--model", str(cli_env["model_path"]),
            ],
            env=cli_env["env"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "Model signed successfully" in result.stdout
        
        # Check signature file exists
        sig_path = cli_env["model_path"].with_suffix(".pth.sig")
        assert sig_path.exists()
        
    def test_cli_verify_command(self, cli_env):
        """Test CLI verify command."""
        import subprocess
        
        # First sign the model
        signer = ModelSigner("cli-test-key")
        signer.sign_model(cli_env["model_path"])
        
        # Then verify via CLI
        result = subprocess.run(
            [
                sys.executable,
                "scripts/sign_model.py",
                "verify",
                "--model", str(cli_env["model_path"]),
            ],
            env=cli_env["env"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "Signature verification PASSED" in result.stdout
        
    def test_cli_verify_failure(self, cli_env):
        """Test CLI verify command with invalid signature."""
        import subprocess
        
        # Sign with different key
        signer = ModelSigner("wrong-key")
        signer.sign_model(cli_env["model_path"])
        
        # Verify should fail
        result = subprocess.run(
            [
                sys.executable,
                "scripts/sign_model.py",
                "verify",
                "--model", str(cli_env["model_path"]),
            ],
            env=cli_env["env"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 1
        assert "Signature verification FAILED" in result.stderr


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_large_file_signing(self, temp_dir):
        """Test signing large files (simulated)."""
        # Create a "large" file (1MB for testing)
        large_file = temp_dir / "large_model.pth"
        large_file.write_bytes(b"x" * (1024 * 1024))
        
        signer = ModelSigner("test-key")
        metadata = signer.sign_model(large_file)
        
        assert metadata["file_size"] == 1024 * 1024
        
        # Verify it
        result, error = signer.verify_model(large_file)
        assert result is True
        
    def test_special_characters_in_filename(self, temp_dir):
        """Test handling files with special characters."""
        # Create file with spaces and special chars
        special_file = temp_dir / "model (v2) [final].pth"
        special_file.write_bytes(b"model data")
        
        signer = ModelSigner("test-key")
        metadata = signer.sign_model(special_file)
        
        assert metadata["file"] == "model (v2) [final].pth"
        
        # Verify it
        result, error = signer.verify_model(special_file)
        assert result is True
        
    def test_concurrent_signing(self, temp_dir):
        """Test that concurrent signing doesn't cause issues."""
        import concurrent.futures
        
        signer = ModelSigner("test-key")
        
        def sign_model(index):
            model_path = temp_dir / f"concurrent_model_{index}.pth"
            model_path.write_bytes(f"model {index}".encode())
            return signer.sign_model(model_path)
            
        # Sign multiple models concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(sign_model, i) for i in range(10)]
            results = [f.result() for f in futures]
            
        assert len(results) == 10
        
        # Verify all signatures
        passed, failed, _ = signer.verify_directory(temp_dir)
        assert passed == 10
        assert failed == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])