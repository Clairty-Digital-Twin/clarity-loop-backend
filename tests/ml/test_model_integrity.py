"""Comprehensive tests for model integrity verification.

Tests cover:
- Model checksum generation and verification
- Model manifest management
- File integrity checking
- Error handling and edge cases
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any
from unittest.mock import patch

import pytest

from clarity.ml.model_integrity import (
    ModelChecksumManager,
    ModelIntegrityError,
    gemini_model_manager,
    pat_model_manager,
    verify_startup_models,
)


@pytest.fixture
def temp_models_dir(tmp_path: Path) -> Path:
    """Create a temporary models directory."""
    models_dir = tmp_path / "test_models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


@pytest.fixture
def sample_model_file(temp_models_dir: Path) -> Path:
    """Create a sample model file."""
    model_file = temp_models_dir / "test_model.bin"
    model_file.write_bytes(b"fake model data for testing")
    return model_file


@pytest.fixture
def sample_model_files(temp_models_dir: Path) -> list[Path]:
    """Create multiple sample model files."""
    files = []
    for i in range(3):
        file_path = temp_models_dir / f"model_{i}.bin"
        file_path.write_bytes(b"fake model data " * (i + 1))
        files.append(file_path)
    return files


@pytest.fixture
def checksum_manager(temp_models_dir: Path) -> ModelChecksumManager:
    """Create a checksum manager with temporary directory."""
    return ModelChecksumManager(temp_models_dir)


@pytest.fixture
def sample_manifest() -> dict[str, Any]:
    """Sample model manifest."""
    return {
        "created_at": "2024-01-01T00:00:00Z",
        "total_size_bytes": 1000,
        "files": {
            "model.bin": {
                "checksum": "abc123def456",
                "size_bytes": 500,
                "algorithm": "sha256",
            },
            "config.json": {
                "checksum": "def456ghi789",
                "size_bytes": 500,
                "algorithm": "sha256",
            },
        },
    }


class TestModelChecksumManager:
    """Test ModelChecksumManager functionality."""

    @staticmethod
    def test_initialization(temp_models_dir: Path) -> None:
        """Test checksum manager initialization."""
        manager = ModelChecksumManager(temp_models_dir)

        assert manager.models_dir == temp_models_dir
        assert manager.checksums_file == temp_models_dir / "checksums.json"
        assert temp_models_dir.exists()

    @staticmethod
    def test_initialization_with_string_path() -> None:
        """Test initialization with string path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ModelChecksumManager(tmp_dir)

            assert manager.models_dir == Path(tmp_dir)
            assert manager.models_dir.exists()

    @staticmethod
    def test_calculate_file_checksum(
        checksum_manager: ModelChecksumManager, sample_model_file: Path
    ) -> None:
        """Test file checksum calculation."""
        checksum = checksum_manager._calculate_file_checksum(sample_model_file)

        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex length

        # Same file should produce same checksum
        checksum2 = checksum_manager._calculate_file_checksum(sample_model_file)
        assert checksum == checksum2

    @staticmethod
    def test_calculate_file_checksum_nonexistent(
        checksum_manager: ModelChecksumManager, temp_models_dir: Path
    ) -> None:
        """Test checksum calculation for non-existent file."""
        nonexistent_file = temp_models_dir / "nonexistent.bin"

        with pytest.raises(ModelIntegrityError, match="Failed to calculate checksum"):
            checksum_manager._calculate_file_checksum(nonexistent_file)

    @staticmethod
    def test_generate_model_manifest_success(
        checksum_manager: ModelChecksumManager, sample_model_files: list[Path]
    ) -> None:
        """Test successful model manifest generation."""
        file_names = [f.name for f in sample_model_files]

        manifest = checksum_manager.generate_model_manifest(file_names)

        assert "created_at" in manifest
        assert "total_size_bytes" in manifest
        assert "files" in manifest
        assert len(manifest["files"]) == len(file_names)

        for file_name in file_names:
            assert file_name in manifest["files"]
            file_info = manifest["files"][file_name]
            assert "checksum" in file_info
            assert "size_bytes" in file_info
            assert "algorithm" in file_info
            assert file_info["algorithm"] == "sha256"

    @staticmethod
    def test_generate_model_manifest_missing_file(
        checksum_manager: ModelChecksumManager,
    ) -> None:
        """Test manifest generation with missing file."""
        with pytest.raises(ModelIntegrityError, match="Model file not found"):
            checksum_manager.generate_model_manifest(["nonexistent.bin"])

    @staticmethod
    def test_save_and_load_checksums(
        checksum_manager: ModelChecksumManager, sample_manifest: dict[str, Any]
    ) -> None:
        """Test saving and loading checksums."""
        manifests = {"test_model": sample_manifest}

        # Save checksums
        checksum_manager.save_checksums(manifests)

        # Verify file was created
        assert checksum_manager.checksums_file.exists()

        # Load checksums
        loaded_manifests = checksum_manager.load_checksums()

        assert loaded_manifests == manifests

    @staticmethod
    def test_load_checksums_missing_file(
        checksum_manager: ModelChecksumManager,
    ) -> None:
        """Test loading checksums when file doesn't exist."""
        result = checksum_manager.load_checksums()

        assert result == {}

    @staticmethod
    def test_load_checksums_invalid_json(
        checksum_manager: ModelChecksumManager,
    ) -> None:
        """Test loading checksums with invalid JSON."""
        # Create invalid JSON file
        checksum_manager.checksums_file.write_text("invalid json content")

        with pytest.raises(ModelIntegrityError, match="Failed to load checksums"):
            checksum_manager.load_checksums()

    @staticmethod
    def test_register_model(
        checksum_manager: ModelChecksumManager, sample_model_files: list[Path]
    ) -> None:
        """Test model registration."""
        file_names = [f.name for f in sample_model_files]
        model_name = "test_model"

        checksum_manager.register_model(model_name, file_names)

        # Verify model was registered
        checksums = checksum_manager.load_checksums()
        assert model_name in checksums
        assert "files" in checksums[model_name]
        assert len(checksums[model_name]["files"]) == len(file_names)

    @staticmethod
    def test_verify_model_integrity_success(
        checksum_manager: ModelChecksumManager, sample_model_files: list[Path]
    ) -> None:
        """Test successful model integrity verification."""
        file_names = [f.name for f in sample_model_files]
        model_name = "test_model"

        # Register model first
        checksum_manager.register_model(model_name, file_names)

        # Verify integrity
        result = checksum_manager.verify_model_integrity(model_name)

        assert result is True

    @staticmethod
    def test_verify_model_integrity_not_registered(
        checksum_manager: ModelChecksumManager,
    ) -> None:
        """Test verification of unregistered model."""
        with pytest.raises(ModelIntegrityError, match="No checksums found for model"):
            checksum_manager.verify_model_integrity("nonexistent_model")

    @staticmethod
    def test_verify_model_integrity_missing_file(
        checksum_manager: ModelChecksumManager, sample_model_files: list[Path]
    ) -> None:
        """Test verification when model file is missing."""
        file_names = [f.name for f in sample_model_files]
        model_name = "test_model"

        # Register model
        checksum_manager.register_model(model_name, file_names)

        # Remove a file
        sample_model_files[0].unlink()

        # Verification should fail
        result = checksum_manager.verify_model_integrity(model_name)

        assert result is False

    @staticmethod
    def test_verify_model_integrity_checksum_mismatch(
        checksum_manager: ModelChecksumManager, sample_model_files: list[Path]
    ) -> None:
        """Test verification with checksum mismatch."""
        file_names = [f.name for f in sample_model_files]
        model_name = "test_model"

        # Register model
        checksum_manager.register_model(model_name, file_names)

        # Modify a file
        sample_model_files[0].write_bytes(b"modified content")

        # Verification should fail
        result = checksum_manager.verify_model_integrity(model_name)

        assert result is False

    @staticmethod
    def test_verify_all_models(
        checksum_manager: ModelChecksumManager, sample_model_files: list[Path]
    ) -> None:
        """Test verification of all models."""
        # Register multiple models
        for i, model_file in enumerate(sample_model_files):
            model_name = f"model_{i}"
            checksum_manager.register_model(model_name, [model_file.name])

        # Verify all models
        results = checksum_manager.verify_all_models()

        assert len(results) == len(sample_model_files)
        assert all(result is True for result in results.values())

    @staticmethod
    def test_get_model_info(
        checksum_manager: ModelChecksumManager, sample_model_files: list[Path]
    ) -> None:
        """Test getting model information."""
        file_names = [f.name for f in sample_model_files]
        model_name = "test_model"

        # Register model
        checksum_manager.register_model(model_name, file_names)

        # Get model info
        info = checksum_manager.get_model_info(model_name)

        assert info is not None
        assert "files" in info
        assert "created_at" in info
        assert len(info["files"]) == len(file_names)

    @staticmethod
    def test_get_model_info_not_found(checksum_manager: ModelChecksumManager) -> None:
        """Test getting info for non-existent model."""
        info = checksum_manager.get_model_info("nonexistent_model")

        assert info is None

    @staticmethod
    def test_list_registered_models(
        checksum_manager: ModelChecksumManager, sample_model_files: list[Path]
    ) -> None:
        """Test listing registered models."""
        model_names = []
        for i, model_file in enumerate(sample_model_files):
            model_name = f"model_{i}"
            model_names.append(model_name)
            checksum_manager.register_model(model_name, [model_file.name])

        registered_models = checksum_manager.list_registered_models()

        assert len(registered_models) == len(model_names)
        assert all(name in registered_models for name in model_names)

    @staticmethod
    def test_remove_model(
        checksum_manager: ModelChecksumManager, sample_model_files: list[Path]
    ) -> None:
        """Test removing a model from registry."""
        model_name = "test_model"

        # Register model
        checksum_manager.register_model(model_name, [sample_model_files[0].name])

        # Verify it exists
        assert model_name in checksum_manager.list_registered_models()

        # Remove model
        checksum_manager.remove_model(model_name)

        # Verify it's gone
        assert model_name not in checksum_manager.list_registered_models()

    @staticmethod
    def test_remove_nonexistent_model(checksum_manager: ModelChecksumManager) -> None:
        """Test removing non-existent model."""
        # Should not raise exception
        checksum_manager.remove_model("nonexistent_model")


class TestModuleGlobals:
    """Test module-level global objects and functions."""

    @staticmethod
    def test_global_managers_exist() -> None:
        """Test that global model managers exist."""
        assert pat_model_manager is not None
        assert gemini_model_manager is not None

        assert isinstance(pat_model_manager, ModelChecksumManager)
        assert isinstance(gemini_model_manager, ModelChecksumManager)

    @staticmethod
    def test_verify_startup_models_no_models() -> None:
        """Test startup verification with no models."""
        with (
            patch.object(pat_model_manager, "verify_all_models", return_value={}),
            patch.object(gemini_model_manager, "verify_all_models", return_value={}),
        ):
            result = verify_startup_models()

            assert result is True

    @staticmethod
    def test_verify_startup_models_all_pass() -> None:
        """Test startup verification when all models pass."""
        with (
            patch.object(
                pat_model_manager,
                "verify_all_models",
                return_value={"model1": True, "model2": True},
            ),
            patch.object(
                gemini_model_manager, "verify_all_models", return_value={"model3": True}
            ),
        ):
            result = verify_startup_models()

            assert result is True

    @staticmethod
    def test_verify_startup_models_some_fail() -> None:
        """Test startup verification when some models fail."""
        with (
            patch.object(
                pat_model_manager,
                "verify_all_models",
                return_value={"model1": True, "model2": False},
            ),
            patch.object(
                gemini_model_manager, "verify_all_models", return_value={"model3": True}
            ),
        ):
            result = verify_startup_models()

            assert result is False

    @staticmethod
    def test_verify_startup_models_exception() -> None:
        """Test startup verification with exception."""
        with (
            patch.object(
                pat_model_manager,
                "verify_all_models",
                side_effect=Exception("Test error"),
            ),
            patch.object(gemini_model_manager, "verify_all_models", return_value={}),
        ):
            result = verify_startup_models()

            assert result is False


class TestErrorHandling:
    """Test error handling scenarios."""

    @staticmethod
    def test_save_checksums_permission_error(
        checksum_manager: ModelChecksumManager,
    ) -> None:
        """Test saving checksums with permission error."""
        # Make directory read-only
        checksum_manager.models_dir.chmod(0o444)

        try:
            with pytest.raises(ModelIntegrityError, match="Failed to save checksums"):
                checksum_manager.save_checksums({"test": {}})
        finally:
            # Restore permissions for cleanup
            checksum_manager.models_dir.chmod(0o755)

    @staticmethod
    def test_generate_manifest_with_permission_error(
        checksum_manager: ModelChecksumManager, sample_model_file: Path
    ) -> None:
        """Test manifest generation with file permission error."""
        # Remove read permissions
        sample_model_file.chmod(0o000)

        try:
            with pytest.raises(ModelIntegrityError, match="Failed to process"):
                checksum_manager.generate_model_manifest([sample_model_file.name])
        finally:
            # Restore permissions for cleanup
            sample_model_file.chmod(0o644)

    @staticmethod
    def test_verify_integrity_with_file_error(
        checksum_manager: ModelChecksumManager, sample_model_files: list[Path]
    ) -> None:
        """Test integrity verification with file access error."""
        model_name = "test_model"

        # Register model
        checksum_manager.register_model(model_name, [sample_model_files[0].name])

        # Remove read permissions from file
        sample_model_files[0].chmod(0o000)

        try:
            # Verification should fail gracefully
            result = checksum_manager.verify_model_integrity(model_name)
            assert result is False
        finally:
            # Restore permissions for cleanup
            sample_model_files[0].chmod(0o644)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @staticmethod
    def test_empty_file_checksum(
        checksum_manager: ModelChecksumManager, temp_models_dir: Path
    ) -> None:
        """Test checksum of empty file."""
        empty_file = temp_models_dir / "empty.bin"
        empty_file.touch()

        checksum = checksum_manager._calculate_file_checksum(empty_file)

        # Should still produce valid checksum
        assert isinstance(checksum, str)
        assert len(checksum) == 64

    @staticmethod
    def test_large_file_handling(
        checksum_manager: ModelChecksumManager, temp_models_dir: Path
    ) -> None:
        """Test handling of larger files (chunked reading)."""
        large_file = temp_models_dir / "large.bin"
        # Create file larger than chunk size
        large_content = b"A" * 10000  # 10KB
        large_file.write_bytes(large_content)

        checksum = checksum_manager._calculate_file_checksum(large_file)

        assert isinstance(checksum, str)
        assert len(checksum) == 64

    @staticmethod
    def test_unicode_file_names(
        checksum_manager: ModelChecksumManager, temp_models_dir: Path
    ) -> None:
        """Test handling of unicode file names."""
        unicode_file = temp_models_dir / "模型文件.bin"
        unicode_file.write_bytes(b"test content")

        checksum = checksum_manager._calculate_file_checksum(unicode_file)

        assert isinstance(checksum, str)
        assert len(checksum) == 64

    @staticmethod
    def test_manifest_with_zero_files(checksum_manager: ModelChecksumManager) -> None:
        """Test generating manifest with empty file list."""
        manifest = checksum_manager.generate_model_manifest([])

        assert manifest["files"] == {}
        assert manifest["total_size_bytes"] == 0
        assert "created_at" in manifest

    @staticmethod
    def test_concurrent_access_simulation(
        checksum_manager: ModelChecksumManager, sample_model_files: list[Path]
    ) -> None:
        """Test behavior with multiple operations (simulating concurrent access)."""
        model_name = "concurrent_test"

        # Simulate multiple operations
        checksum_manager.register_model(model_name, [sample_model_files[0].name])

        # Multiple reads should work
        for _ in range(3):
            info = checksum_manager.get_model_info(model_name)
            assert info is not None

            result = checksum_manager.verify_model_integrity(model_name)
            assert result is True
