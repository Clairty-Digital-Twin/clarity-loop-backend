"""ML Model Integrity Verification System.

Provides checksum verification and integrity checking for ML model weights
following security best practices for healthcare AI systems.
"""

# removed - breaks FastAPI

from datetime import UTC, datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Constants
MAX_CHUNK_SIZE = 8192
DEFAULT_ENCODING = "utf-8"


class ModelIntegrityError(Exception):
    """Raised when model integrity verification fails."""


class ModelChecksumManager:
    """Manages checksum generation and verification for ML model files.

    Provides SHA-256 checksums for model weights and metadata to ensure
    model integrity in healthcare AI applications.
    """

    def __init__(self, models_dir: Path | str = "models") -> None:
        """Initialize the checksum manager.

        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.checksums_file = self.models_dir / "checksums.json"

        # Ensure models directory exists
        self.models_dir.mkdir(exist_ok=True)

    @staticmethod
    def _calculate_file_checksum(file_path: Path) -> str:
        """Calculate SHA-256 checksum for a file.

        Args:
            file_path: Path to the file

        Returns:
            Hexadecimal SHA-256 checksum

        Raises:
            ModelIntegrityError: If file cannot be read
        """
        try:
            sha256_hash = hashlib.sha256()
            with file_path.open("rb") as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(MAX_CHUNK_SIZE), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except OSError as e:
            error_msg = f"Failed to calculate checksum for {file_path}: {e}"
            logger.exception("Error calculating file checksum")
            raise ModelIntegrityError(error_msg) from e

    def generate_model_manifest(self, model_files: list[str]) -> dict[str, Any]:
        """Generate a manifest with checksums for model files.

        Args:
            model_files: List of model file names in the models directory

        Returns:
            Manifest dictionary with file checksums and metadata

        Raises:
            ModelIntegrityError: If any file cannot be processed
        """
        manifest: dict[str, Any] = {
            "created_at": datetime.now(UTC).isoformat(),
            "total_size_bytes": 0,
            "files": {},
        }

        for file_name in model_files:
            file_path = self.models_dir / file_name

            if not file_path.exists():
                error_msg = f"Model file not found: {file_path}"
                logger.error("Model file missing: %s", file_path)
                raise ModelIntegrityError(error_msg)

            try:
                checksum = self._calculate_file_checksum(file_path)
                file_size = file_path.stat().st_size

                manifest["files"][file_name] = {
                    "checksum": checksum,
                    "size_bytes": file_size,
                    "algorithm": "sha256",
                }

                manifest["total_size_bytes"] += file_size

                logger.info("Generated checksum for %s: %s", file_name, checksum)

            except Exception as e:
                logger.exception("Error processing file: %s", file_name)
                error_msg = f"Failed to process {file_name}: {e}"
                raise ModelIntegrityError(error_msg) from e

        return manifest

    def save_checksums(self, manifests: dict[str, dict[str, Any]]) -> None:
        """Save model manifests to the checksums file.

        Args:
            manifests: Dictionary mapping model names to their manifests

        Raises:
            ModelIntegrityError: If checksums cannot be saved
        """
        try:
            with self.checksums_file.open("w", encoding=DEFAULT_ENCODING) as f:
                json.dump(manifests, f, indent=2, sort_keys=True)

            logger.info(
                "Saved checksums for %d models to %s",
                len(manifests),
                self.checksums_file,
            )

        except OSError as e:
            error_msg = f"Failed to save checksums: {e}"
            raise ModelIntegrityError(error_msg) from e

    def load_checksums(self) -> dict[str, dict[str, Any]]:
        """Load model manifests from the checksums file.

        Returns:
            Dictionary mapping model names to their manifests

        Raises:
            ModelIntegrityError: If checksums cannot be loaded
        """
        if not self.checksums_file.exists():
            logger.warning("Checksums file not found: %s", self.checksums_file)
            return {}

        try:
            with self.checksums_file.open(encoding=DEFAULT_ENCODING) as f:
                data: dict[str, dict[str, Any]] = json.load(f)
                return data
        except (OSError, json.JSONDecodeError) as e:
            error_msg = f"Failed to load checksums: {e}"
            raise ModelIntegrityError(error_msg) from e

    def verify_model_integrity(self, model_name: str) -> bool:
        """Verify the integrity of a specific model.

        Args:
            model_name: Name of the model to verify

        Returns:
            True if all model files pass verification, False otherwise

        Raises:
            ModelIntegrityError: If model is not registered or verification fails
        """
        checksums = self.load_checksums()

        if model_name not in checksums:
            error_msg = f"No checksums found for model: {model_name}"
            raise ModelIntegrityError(error_msg)

        manifest = checksums[model_name]
        verification_results = []

        logger.info("Verifying integrity of model: %s", model_name)

        for file_name, file_info in manifest["files"].items():
            file_path = self.models_dir / file_name
            expected_checksum = file_info["checksum"]

            if not file_path.exists():
                logger.error("Model file missing: %s", file_path)
                verification_results.append(False)
                continue

            try:
                actual_checksum = self._calculate_file_checksum(file_path)

                if actual_checksum == expected_checksum:
                    logger.debug("✓ %s: checksum verified", file_name)
                    verification_results.append(True)
                else:
                    logger.error(
                        "✗ %s: checksum mismatch! Expected: %s, Got: %s",
                        file_name,
                        expected_checksum,
                        actual_checksum,
                    )
                    verification_results.append(False)

            except Exception:
                logger.exception("Error verifying file: %s", file_name)
                verification_results.append(False)

        all_verified = all(verification_results)

        if all_verified:
            logger.info("✓ Model %s integrity verification PASSED", model_name)
        else:
            logger.error("✗ Model %s integrity verification FAILED", model_name)

        return all_verified

    def verify_all_models(self) -> dict[str, bool]:
        """Verify the integrity of all registered models.

        Returns:
            Dictionary mapping model names to verification results
        """
        checksums = self.load_checksums()
        results = {}

        for model_name in checksums:
            try:
                results[model_name] = self.verify_model_integrity(model_name)
            except Exception:
                logger.exception("Error verifying model: %s", model_name)
                results[model_name] = False

        return results

    def register_model(self, model_name: str, model_files: list[str]) -> None:
        """Register a new model with checksum generation.

        Args:
            model_name: Unique name for the model
            model_files: List of model files to register
        """
        logger.info("Registering model: %s", model_name)

        # Generate manifest for the new model
        manifest = self.generate_model_manifest(model_files)

        # Load existing checksums and add the new model
        checksums = self.load_checksums()
        checksums[model_name] = manifest

        # Save updated checksums
        self.save_checksums(checksums)

        logger.info("Successfully registered model: %s", model_name)

    def get_model_info(self, model_name: str) -> dict[str, Any] | None:
        """Get information about a registered model.

        Args:
            model_name: Name of the model

        Returns:
            Model manifest dictionary or None if not found
        """
        checksums = self.load_checksums()
        return checksums.get(model_name)

    def list_registered_models(self) -> list[str]:
        """Get a list of all registered model names.

        Returns:
            List of registered model names
        """
        checksums = self.load_checksums()
        return list(checksums.keys())

    def remove_model(self, model_name: str) -> None:
        """Remove a model from the registry.

        Args:
            model_name: Name of the model to remove
        """
        checksums = self.load_checksums()

        if model_name in checksums:
            del checksums[model_name]
            self.save_checksums(checksums)
            logger.info("Removed model from registry: %s", model_name)
        else:
            logger.warning("Model not found in registry: %s", model_name)


# Global model managers for different model types
pat_model_manager = ModelChecksumManager("models/pat")
gemini_model_manager = ModelChecksumManager("models/gemini")


def verify_startup_models() -> bool:
    """Verify integrity of all critical models during application startup.

    Returns:
        True if all critical models pass verification, False otherwise
    """
    model_managers = [
        ("PAT models", pat_model_manager),
        ("Gemini models", gemini_model_manager),
    ]

    all_passed = True

    for name, manager in model_managers:
        try:
            results = manager.verify_all_models()

            if not results:
                logger.info("No %s found to verify", name)
                continue

            passed_count = sum(results.values())
            total_count = len(results)

            if passed_count == total_count:
                logger.info("✓ All %s (%d) passed verification", name, total_count)
            else:
                logger.error(
                    "✗ %s: %d/%d models passed verification",
                    name,
                    passed_count,
                    total_count,
                )
                all_passed = False

        except Exception:
            logger.exception("Error verifying %s", name)
            all_passed = False

    return all_passed
