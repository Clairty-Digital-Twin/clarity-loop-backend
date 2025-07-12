#!/usr/bin/env python3
"""Model verification for application startup.

This script demonstrates how to integrate model signature verification
into your application's startup process. It ensures all models are
properly signed and verified before the application starts.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def verify_model_signatures(models_dir: str = "models/") -> Tuple[bool, List[str]]:
    """Verify all model signatures in the specified directory.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        Tuple of (success, error_messages)
    """
    logger.info("Starting model signature verification...")
    
    # Check if signing key is available
    if not os.environ.get("MODEL_SIGNING_KEY"):
        logger.warning(
            "MODEL_SIGNING_KEY not set. Skipping signature verification. "
            "This should only happen in development mode."
        )
        return True, []
        
    # Run the signature verification
    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/sign_model.py",
                "verify-all",
                "--models-dir", models_dir,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        
        if result.returncode == 0:
            logger.info("✓ All model signatures verified successfully")
            return True, []
        else:
            error_messages = result.stderr.strip().split("\n")
            logger.error("✗ Model signature verification failed")
            for msg in error_messages:
                if msg.strip():
                    logger.error("  %s", msg)
            return False, error_messages
            
    except Exception as e:
        error_msg = f"Error running signature verification: {e}"
        logger.exception(error_msg)
        return False, [error_msg]


def verify_model_checksums() -> bool:
    """Verify model checksums using the existing integrity system.
    
    Returns:
        True if all checksums pass, False otherwise
    """
    logger.info("Starting model checksum verification...")
    
    try:
        # Import the existing integrity verification
        from clarity.ml.model_integrity import verify_startup_models
        
        success = verify_startup_models()
        if success:
            logger.info("✓ All model checksums verified successfully")
        else:
            logger.error("✗ Model checksum verification failed")
            
        return success
        
    except ImportError:
        logger.warning("Model integrity module not found. Skipping checksum verification.")
        return True
    except Exception as e:
        logger.exception("Error during checksum verification: %s", e)
        return False


def check_model_files_exist(models_dir: str = "models/") -> Tuple[bool, List[str]]:
    """Check that expected model files exist.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        Tuple of (all_exist, missing_files)
    """
    # Define expected model files (customize for your application)
    expected_models = [
        "pat/model.pth",
        "gemini/model.onnx",
        # Add your expected model files here
    ]
    
    models_path = Path(models_dir)
    missing_files = []
    
    for model_file in expected_models:
        full_path = models_path / model_file
        if not full_path.exists():
            missing_files.append(model_file)
            logger.error("Missing model file: %s", full_path)
            
    if missing_files:
        return False, missing_files
    else:
        logger.info("✓ All expected model files exist")
        return True, []


def startup_verification(
    require_signatures: bool = True,
    require_checksums: bool = True,
    models_dir: str = "models/"
) -> bool:
    """Perform complete model verification for application startup.
    
    Args:
        require_signatures: Whether to require valid signatures
        require_checksums: Whether to require valid checksums
        models_dir: Directory containing model files
        
    Returns:
        True if all verifications pass, False otherwise
    """
    logger.info("=" * 60)
    logger.info("Model Verification for Application Startup")
    logger.info("=" * 60)
    
    all_passed = True
    
    # Step 1: Check model files exist
    files_exist, missing = check_model_files_exist(models_dir)
    if not files_exist:
        logger.error("Cannot start: %d model files are missing", len(missing))
        all_passed = False
        
    # Step 2: Verify signatures (if required)
    if require_signatures and all_passed:
        sig_passed, sig_errors = verify_model_signatures(models_dir)
        if not sig_passed:
            logger.error("Cannot start: Model signature verification failed")
            all_passed = False
            
    # Step 3: Verify checksums (if required)
    if require_checksums and all_passed:
        checksum_passed = verify_model_checksums()
        if not checksum_passed:
            logger.error("Cannot start: Model checksum verification failed")
            all_passed = False
            
    # Summary
    logger.info("=" * 60)
    if all_passed:
        logger.info("✓ All model verifications PASSED")
        logger.info("✓ Application can start safely")
    else:
        logger.error("✗ Model verification FAILED")
        logger.error("✗ Application startup blocked for safety")
        
    return all_passed


def main():
    """Main entry point for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify model signatures and integrity for application startup"
    )
    parser.add_argument(
        "--models-dir",
        default="models/",
        help="Directory containing model files (default: models/)",
    )
    parser.add_argument(
        "--no-signatures",
        action="store_true",
        help="Skip signature verification",
    )
    parser.add_argument(
        "--no-checksums",
        action="store_true",
        help="Skip checksum verification",
    )
    parser.add_argument(
        "--development",
        action="store_true",
        help="Development mode (warnings instead of errors)",
    )
    
    args = parser.parse_args()
    
    # In development mode, don't require signatures
    if args.development:
        logger.info("Running in DEVELOPMENT mode - verification failures will be warnings only")
        
    success = startup_verification(
        require_signatures=not args.no_signatures and not args.development,
        require_checksums=not args.no_checksums,
        models_dir=args.models_dir,
    )
    
    if not success and not args.development:
        sys.exit(1)
        
        
# Example integration into your FastAPI/Flask application:
#
# from scripts.verify_models_startup import startup_verification
#
# # In your main application file
# if not startup_verification():
#     print("Model verification failed. Exiting.")
#     sys.exit(1)
#
# # Continue with normal application startup
# app = FastAPI()
# ...


if __name__ == "__main__":
    main()