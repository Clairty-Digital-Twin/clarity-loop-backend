#!/usr/bin/env python3
"""Model Signing and Verification Tool.

Provides HMAC-based signing for ML model files to ensure integrity and authenticity.
Designed for CI/CD integration with support for both signing and verification modes.

Usage:
    # Sign a model file
    python sign_model.py sign --model models/pat/model.pth --key $SECRET_KEY
    
    # Verify a model signature
    python sign_model.py verify --model models/pat/model.pth --key $SECRET_KEY
    
    # Sign all models in a directory
    python sign_model.py sign-all --models-dir models/ --key $SECRET_KEY
    
    # Verify all models in a directory
    python sign_model.py verify-all --models-dir models/ --key $SECRET_KEY
"""

import argparse
import hashlib
import hmac
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
SIGNATURE_EXTENSION = ".sig"
MANIFEST_FILE = "model_signatures.json"
ALGORITHM = "sha256"
CHUNK_SIZE = 8192
DEFAULT_ENCODING = "utf-8"


class ModelSigningError(Exception):
    """Raised when model signing or verification fails."""


class ModelSigner:
    """Handles HMAC-based signing and verification of ML model files."""

    def __init__(self, secret_key: str) -> None:
        """Initialize the model signer with a secret key.
        
        Args:
            secret_key: Secret key for HMAC signing
        """
        if not secret_key:
            raise ValueError("Secret key cannot be empty")
        
        self.secret_key = secret_key.encode(DEFAULT_ENCODING)
        
    def _calculate_hmac(self, file_path: Path) -> str:
        """Calculate HMAC-SHA256 for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hexadecimal HMAC signature
            
        Raises:
            ModelSigningError: If file cannot be read
        """
        try:
            h = hmac.new(self.secret_key, digestmod=hashlib.sha256)
            
            with file_path.open("rb") as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                    h.update(chunk)
                    
            return h.hexdigest()
            
        except OSError as e:
            error_msg = f"Failed to calculate HMAC for {file_path}: {e}"
            logger.exception("Error calculating file HMAC")
            raise ModelSigningError(error_msg) from e
            
    def sign_model(self, model_path: Path) -> Dict[str, Any]:
        """Sign a model file and create a signature file.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Signature metadata dictionary
            
        Raises:
            ModelSigningError: If signing fails
        """
        if not model_path.exists():
            raise ModelSigningError(f"Model file not found: {model_path}")
            
        logger.info("Signing model: %s", model_path)
        
        # Calculate HMAC signature
        signature = self._calculate_hmac(model_path)
        
        # Create signature metadata
        metadata = {
            "file": str(model_path.name),
            "signature": signature,
            "algorithm": ALGORITHM,
            "signed_at": datetime.now(UTC).isoformat(),
            "file_size": model_path.stat().st_size,
        }
        
        # Save signature file
        sig_path = model_path.with_suffix(model_path.suffix + SIGNATURE_EXTENSION)
        try:
            with sig_path.open("w", encoding=DEFAULT_ENCODING) as f:
                json.dump(metadata, f, indent=2)
                
            logger.info("✓ Signature saved to: %s", sig_path)
            
        except OSError as e:
            raise ModelSigningError(f"Failed to save signature: {e}") from e
            
        return metadata
        
    def verify_model(self, model_path: Path) -> Tuple[bool, Optional[str]]:
        """Verify a model file against its signature.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Tuple of (verification_result, error_message)
            
        Raises:
            ModelSigningError: If verification process fails
        """
        if not model_path.exists():
            return False, f"Model file not found: {model_path}"
            
        # Look for signature file
        sig_path = model_path.with_suffix(model_path.suffix + SIGNATURE_EXTENSION)
        if not sig_path.exists():
            return False, f"Signature file not found: {sig_path}"
            
        logger.info("Verifying model: %s", model_path)
        
        try:
            # Load signature metadata
            with sig_path.open(encoding=DEFAULT_ENCODING) as f:
                metadata = json.load(f)
                
            expected_signature = metadata.get("signature")
            if not expected_signature:
                return False, "Invalid signature file: missing signature"
                
            # Verify file size hasn't changed
            current_size = model_path.stat().st_size
            expected_size = metadata.get("file_size")
            if expected_size and current_size != expected_size:
                return False, f"File size mismatch: expected {expected_size}, got {current_size}"
                
            # Calculate current HMAC
            actual_signature = self._calculate_hmac(model_path)
            
            # Compare signatures
            if hmac.compare_digest(actual_signature, expected_signature):
                logger.info("✓ Signature verified successfully")
                return True, None
            else:
                return False, "Signature mismatch"
                
        except (OSError, json.JSONDecodeError) as e:
            raise ModelSigningError(f"Failed to verify signature: {e}") from e
            
    def sign_directory(self, directory: Path, recursive: bool = True) -> Dict[str, Dict[str, Any]]:
        """Sign all model files in a directory.
        
        Args:
            directory: Directory containing model files
            recursive: Whether to sign files recursively
            
        Returns:
            Dictionary mapping file paths to signature metadata
        """
        if not directory.exists():
            raise ModelSigningError(f"Directory not found: {directory}")
            
        logger.info("Signing models in directory: %s", directory)
        
        # Define model file patterns
        model_patterns = ["*.pth", "*.pt", "*.onnx", "*.pb", "*.h5", "*.safetensors"]
        
        signatures = {}
        model_files = []
        
        # Find all model files
        for pattern in model_patterns:
            if recursive:
                model_files.extend(directory.rglob(pattern))
            else:
                model_files.extend(directory.glob(pattern))
                
        if not model_files:
            logger.warning("No model files found in %s", directory)
            return signatures
            
        # Sign each model file
        for model_path in model_files:
            try:
                metadata = self.sign_model(model_path)
                relative_path = model_path.relative_to(directory)
                signatures[str(relative_path)] = metadata
                
            except Exception as e:
                logger.error("Failed to sign %s: %s", model_path, e)
                
        # Save manifest file
        manifest_path = directory / MANIFEST_FILE
        try:
            manifest = {
                "created_at": datetime.now(UTC).isoformat(),
                "algorithm": ALGORITHM,
                "total_files": len(signatures),
                "signatures": signatures,
            }
            
            with manifest_path.open("w", encoding=DEFAULT_ENCODING) as f:
                json.dump(manifest, f, indent=2)
                
            logger.info("✓ Manifest saved to: %s", manifest_path)
            
        except OSError as e:
            logger.error("Failed to save manifest: %s", e)
            
        logger.info("✓ Signed %d model files", len(signatures))
        return signatures
        
    def verify_directory(self, directory: Path, recursive: bool = True) -> Tuple[int, int, List[str]]:
        """Verify all model files in a directory.
        
        Args:
            directory: Directory containing model files
            recursive: Whether to verify files recursively
            
        Returns:
            Tuple of (passed_count, failed_count, failed_files)
        """
        if not directory.exists():
            raise ModelSigningError(f"Directory not found: {directory}")
            
        logger.info("Verifying models in directory: %s", directory)
        
        # Define model file patterns
        model_patterns = ["*.pth", "*.pt", "*.onnx", "*.pb", "*.h5", "*.safetensors"]
        
        passed = 0
        failed = 0
        failed_files = []
        model_files = []
        
        # Find all model files
        for pattern in model_patterns:
            if recursive:
                model_files.extend(directory.rglob(pattern))
            else:
                model_files.extend(directory.glob(pattern))
                
        if not model_files:
            logger.warning("No model files found in %s", directory)
            return 0, 0, []
            
        # Verify each model file
        for model_path in model_files:
            try:
                result, error = self.verify_model(model_path)
                if result:
                    passed += 1
                    logger.info("✓ %s: PASSED", model_path.relative_to(directory))
                else:
                    failed += 1
                    failed_files.append(str(model_path.relative_to(directory)))
                    logger.error("✗ %s: FAILED - %s", model_path.relative_to(directory), error)
                    
            except Exception as e:
                failed += 1
                failed_files.append(str(model_path.relative_to(directory)))
                logger.error("✗ %s: ERROR - %s", model_path.relative_to(directory), e)
                
        return passed, failed, failed_files


def get_secret_key() -> str:
    """Get the secret key from environment or prompt.
    
    Returns:
        Secret key string
        
    Raises:
        ValueError: If no secret key is provided
    """
    # Try environment variable first
    key = os.environ.get("MODEL_SIGNING_KEY")
    if key:
        return key
        
    # For CI/CD, fail if no environment variable
    if os.environ.get("CI"):
        raise ValueError("MODEL_SIGNING_KEY environment variable not set")
        
    # Interactive mode: prompt for key
    import getpass
    key = getpass.getpass("Enter signing key: ")
    if not key:
        raise ValueError("No signing key provided")
        
    return key


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Model Signing and Verification Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  MODEL_SIGNING_KEY    Secret key for HMAC signing (recommended for CI/CD)

Examples:
  # Sign a single model file
  python sign_model.py sign --model models/pat/model.pth
  
  # Sign with explicit key (not recommended, use env var instead)
  python sign_model.py sign --model models/pat/model.pth --key mysecretkey
  
  # Verify a model signature
  python sign_model.py verify --model models/pat/model.pth
  
  # Sign all models in a directory recursively
  python sign_model.py sign-all --models-dir models/
  
  # Verify all models in a directory
  python sign_model.py verify-all --models-dir models/
  
  # CI/CD usage with environment variable
  export MODEL_SIGNING_KEY="your-secret-key"
  python sign_model.py sign-all --models-dir models/
        """,
    )
    
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Sign command
    sign_parser = subparsers.add_parser("sign", help="Sign a model file")
    sign_parser.add_argument(
        "--model", required=True, type=Path, help="Path to the model file"
    )
    sign_parser.add_argument(
        "--key", help="Signing key (use MODEL_SIGNING_KEY env var instead)"
    )
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a model signature")
    verify_parser.add_argument(
        "--model", required=True, type=Path, help="Path to the model file"
    )
    verify_parser.add_argument(
        "--key", help="Signing key (use MODEL_SIGNING_KEY env var instead)"
    )
    
    # Sign-all command
    sign_all_parser = subparsers.add_parser("sign-all", help="Sign all models in a directory")
    sign_all_parser.add_argument(
        "--models-dir", required=True, type=Path, help="Directory containing model files"
    )
    sign_all_parser.add_argument(
        "--key", help="Signing key (use MODEL_SIGNING_KEY env var instead)"
    )
    sign_all_parser.add_argument(
        "--no-recursive", action="store_true", help="Don't sign files recursively"
    )
    
    # Verify-all command
    verify_all_parser = subparsers.add_parser("verify-all", help="Verify all models in a directory")
    verify_all_parser.add_argument(
        "--models-dir", required=True, type=Path, help="Directory containing model files"
    )
    verify_all_parser.add_argument(
        "--key", help="Signing key (use MODEL_SIGNING_KEY env var instead)"
    )
    verify_all_parser.add_argument(
        "--no-recursive", action="store_true", help="Don't verify files recursively"
    )
    
    return parser


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    try:
        # Get the secret key
        if hasattr(args, "key") and args.key:
            secret_key = args.key
        else:
            secret_key = get_secret_key()
            
        # Create signer
        signer = ModelSigner(secret_key)
        
        # Execute command
        if args.command == "sign":
            metadata = signer.sign_model(args.model)
            logger.info("✓ Model signed successfully")
            logger.info("Signature: %s", metadata["signature"])
            
        elif args.command == "verify":
            result, error = signer.verify_model(args.model)
            if result:
                logger.info("✓ Signature verification PASSED")
                sys.exit(0)
            else:
                logger.error("✗ Signature verification FAILED: %s", error)
                sys.exit(1)
                
        elif args.command == "sign-all":
            signatures = signer.sign_directory(
                args.models_dir, 
                recursive=not args.no_recursive
            )
            logger.info("✓ Signed %d model files", len(signatures))
            
        elif args.command == "verify-all":
            passed, failed, failed_files = signer.verify_directory(
                args.models_dir,
                recursive=not args.no_recursive
            )
            
            logger.info("\nVerification Summary:")
            logger.info("  Passed: %d", passed)
            logger.info("  Failed: %d", failed)
            
            if failed > 0:
                logger.error("\nFailed files:")
                for file in failed_files:
                    logger.error("  - %s", file)
                sys.exit(1)
            else:
                logger.info("✓ All models verified successfully")
                
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Error: %s", e)
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()