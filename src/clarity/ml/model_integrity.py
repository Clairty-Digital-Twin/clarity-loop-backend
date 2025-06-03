"""ML Model Integrity Verification System.

Provides checksum verification and integrity checking for ML model weights
following security best practices for healthcare AI systems.
"""

from datetime import UTC, datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ModelIntegrityError(Exception):
    """Raised when model integrity verification fails."""


class ModelChecksumManager:
    """Manages checksum generation and verification for ML model files.

    Provides SHA-256 checksums for model weights and metadata to ensure
    model integrity in healthcare AI applications.
    """

    def __init__(self, models_dir: Path | str = "models"):
        """Initialize the checksum manager.

        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.checksums_file = self.models_dir / "model_checksums.json"

        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum for a file.

        Args:
            file_path: Path to the file

        Returns:
            Hexadecimal SHA-256 checksum
        """
        sha256_hash = hashlib.sha256()

        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large model files
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)

            return sha256_hash.hexdigest()
        except FileNotFoundError:
            raise ModelIntegrityError(f"Model file not found: {file_path}")
        except Exception as e:
            raise ModelIntegrityError(
                f"Error calculating checksum for {file_path}: {e}"
            )

    def generate_model_manifest(
        self, model_name: str, model_files: list[str]
    ) -> dict[str, Any]:
        """Generate a complete manifest with checksums for a model.

        Args:
            model_name: Name/identifier of the model
            model_files: List of model files to include in manifest

        Returns:
            Model manifest with checksums and metadata
        """
        manifest = {
            "model_name": model_name,
            "created_at": datetime.now(UTC).isoformat(),
            "files": {},
            "total_files": 0,
            "total_size_bytes": 0,
        }

        for file_name in model_files:
            file_path = self.models_dir / file_name

            if not file_path.exists():
                logger.warning(f"Model file not found: {file_path}")
                continue

            try:
                checksum = self.calculate_file_checksum(file_path)
                file_size = file_path.stat().st_size

                manifest["files"][file_name] = {
                    "checksum": checksum,
                    "size_bytes": file_size,
                    "last_modified": datetime.fromtimestamp(
                        file_path.stat().st_mtime, tz=UTC
                    ).isoformat(),
                }

                manifest["total_files"] += 1
                manifest["total_size_bytes"] += file_size

                logger.info(f"Generated checksum for {file_name}: {checksum}")

            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
                raise ModelIntegrityError(f"Failed to process {file_name}: {e}")

        return manifest

    def save_checksums(self, manifests: dict[str, dict[str, Any]]) -> None:
        """Save model checksums to the checksums file.

        Args:
            manifests: Dictionary of model manifests keyed by model name
        """
        try:
            with open(self.checksums_file, "w") as f:
                json.dump(manifests, f, indent=2, sort_keys=True)

            logger.info(
                f"Saved checksums for {len(manifests)} models to {self.checksums_file}"
            )

        except Exception as e:
            raise ModelIntegrityError(f"Failed to save checksums: {e}")

    def load_checksums(self) -> dict[str, dict[str, Any]]:
        """Load model checksums from the checksums file.

        Returns:
            Dictionary of model manifests keyed by model name
        """
        if not self.checksums_file.exists():
            logger.warning(f"Checksums file not found: {self.checksums_file}")
            return {}

        try:
            with open(self.checksums_file) as f:
                return json.load(f)
        except Exception as e:
            raise ModelIntegrityError(f"Failed to load checksums: {e}")

    def verify_model_integrity(self, model_name: str) -> bool:
        """Verify the integrity of a model using stored checksums.

        Args:
            model_name: Name of the model to verify

        Returns:
            True if all checksums match, False otherwise

        Raises:
            ModelIntegrityError: If verification fails due to errors
        """
        checksums = self.load_checksums()

        if model_name not in checksums:
            raise ModelIntegrityError(f"No checksums found for model: {model_name}")

        manifest = checksums[model_name]
        verification_results = []

        logger.info(f"Verifying integrity of model: {model_name}")

        for file_name, file_info in manifest["files"].items():
            file_path = self.models_dir / file_name
            expected_checksum = file_info["checksum"]

            if not file_path.exists():
                logger.error(f"Model file missing: {file_path}")
                verification_results.append(False)
                continue

            try:
                actual_checksum = self.calculate_file_checksum(file_path)

                if actual_checksum == expected_checksum:
                    logger.debug(f"✓ {file_name}: checksum verified")
                    verification_results.append(True)
                else:
                    logger.error(
                        f"✗ {file_name}: checksum mismatch! "
                        f"Expected: {expected_checksum}, Got: {actual_checksum}"
                    )
                    verification_results.append(False)

            except Exception as e:
                logger.error(f"Error verifying {file_name}: {e}")
                verification_results.append(False)

        all_verified = all(verification_results)

        if all_verified:
            logger.info(f"✓ Model {model_name} integrity verification PASSED")
        else:
            logger.error(f"✗ Model {model_name} integrity verification FAILED")

        return all_verified

    def verify_all_models(self) -> dict[str, bool]:
        """Verify integrity of all models with stored checksums.

        Returns:
            Dictionary mapping model names to verification results
        """
        checksums = self.load_checksums()
        results = {}

        for model_name in checksums:
            try:
                results[model_name] = self.verify_model_integrity(model_name)
            except Exception as e:
                logger.error(f"Error verifying model {model_name}: {e}")
                results[model_name] = False

        return results

    def register_model(self, model_name: str, model_files: list[str]) -> None:
        """Register a new model and generate its checksums.

        Args:
            model_name: Name/identifier of the model
            model_files: List of model files to register
        """
        logger.info(f"Registering model: {model_name}")

        # Generate manifest for the new model
        manifest = self.generate_model_manifest(model_name, model_files)

        # Load existing checksums and add the new model
        checksums = self.load_checksums()
        checksums[model_name] = manifest

        # Save updated checksums
        self.save_checksums(checksums)

        logger.info(f"Successfully registered model: {model_name}")

    def get_model_info(self, model_name: str) -> dict[str, Any] | None:
        """Get information about a registered model.

        Args:
            model_name: Name of the model

        Returns:
            Model information dictionary or None if not found
        """
        checksums = self.load_checksums()
        return checksums.get(model_name)

    def list_registered_models(self) -> list[str]:
        """Get list of all registered model names.

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
            logger.info(f"Removed model from registry: {model_name}")
        else:
            logger.warning(f"Model not found in registry: {model_name}")


# Pre-configured instances for common use cases
pat_model_manager = ModelChecksumManager("models/pat")
gemini_model_manager = ModelChecksumManager("models/gemini")


def verify_startup_models() -> bool:
    """Verify integrity of all critical models during application startup.

    Returns:
        True if all models pass verification, False otherwise
    """
    logger.info("Starting model integrity verification...")

    managers = [
        ("PAT Models", pat_model_manager),
        ("Gemini Models", gemini_model_manager),
    ]

    all_passed = True

    for name, manager in managers:
        try:
            results = manager.verify_all_models()

            if not results:
                logger.info(f"No {name} found to verify")
                continue

            passed_count = sum(results.values())
            total_count = len(results)

            if passed_count == total_count:
                logger.info(f"✓ All {name} ({total_count}) passed verification")
            else:
                logger.error(
                    f"✗ {name}: {passed_count}/{total_count} models passed verification"
                )
                all_passed = False

        except Exception as e:
            logger.error(f"Error verifying {name}: {e}")
            all_passed = False

    if all_passed:
        logger.info("✓ All model integrity checks PASSED")
    else:
        logger.error("✗ Model integrity verification FAILED")

    return all_passed
