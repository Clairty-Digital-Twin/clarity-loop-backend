#!/usr/bin/env python3
"""Model Integrity CLI Tool.

Command-line interface for managing ML model checksums and integrity verification
in the Clarity Loop Backend healthcare AI system.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clarity.ml.model_integrity import (
    ModelChecksumManager,
    ModelIntegrityError,
    pat_model_manager,
    gemini_model_manager,
    verify_startup_models,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def register_model_command(args: argparse.Namespace) -> None:
    """Register a new model with checksum generation."""
    manager = ModelChecksumManager(args.models_dir)
    
    try:
        # Get list of model files
        models_dir = Path(args.models_dir)
        if args.files:
            model_files = args.files
        else:
            # Auto-discover model files
            model_files = []
            for pattern in ["*.pth", "*.pkl", "*.joblib", "*.h5", "*.onnx"]:
                model_files.extend([f.name for f in models_dir.glob(pattern)])
            
            if not model_files:
                logger.error(f"No model files found in {models_dir}")
                return
        
        logger.info(f"Registering model '{args.model_name}' with files: {model_files}")
        manager.register_model(args.model_name, model_files)
        logger.info("✓ Model registered successfully")
        
    except ModelIntegrityError as e:
        logger.error(f"Failed to register model: {e}")
        sys.exit(1)


def verify_model_command(args: argparse.Namespace) -> None:
    """Verify model integrity."""
    manager = ModelChecksumManager(args.models_dir)
    
    try:
        if args.model_name:
            # Verify specific model
            result = manager.verify_model_integrity(args.model_name)
            if result:
                logger.info(f"✓ Model '{args.model_name}' verification PASSED")
            else:
                logger.error(f"✗ Model '{args.model_name}' verification FAILED")
                sys.exit(1)
        else:
            # Verify all models
            results = manager.verify_all_models()
            
            if not results:
                logger.info("No models found to verify")
                return
            
            passed_count = sum(results.values())
            total_count = len(results)
            
            for model_name, passed in results.items():
                status = "✓ PASSED" if passed else "✗ FAILED"
                logger.info(f"{model_name}: {status}")
            
            if passed_count == total_count:
                logger.info(f"✓ All models ({total_count}) verification PASSED")
            else:
                logger.error(f"✗ Verification: {passed_count}/{total_count} models passed")
                sys.exit(1)
                
    except ModelIntegrityError as e:
        logger.error(f"Verification failed: {e}")
        sys.exit(1)


def list_models_command(args: argparse.Namespace) -> None:
    """List registered models."""
    manager = ModelChecksumManager(args.models_dir)
    
    try:
        models = manager.list_registered_models()
        
        if not models:
            logger.info("No models registered")
            return
        
        logger.info(f"Registered models in {args.models_dir}:")
        for model_name in models:
            info = manager.get_model_info(model_name)
            if info:
                file_count = info.get("total_files", 0)
                size_mb = info.get("total_size_bytes", 0) / (1024 * 1024)
                created = info.get("created_at", "unknown")
                logger.info(f"  {model_name}: {file_count} files, {size_mb:.1f}MB, created {created}")
            else:
                logger.info(f"  {model_name}: (info unavailable)")
                
    except ModelIntegrityError as e:
        logger.error(f"Failed to list models: {e}")
        sys.exit(1)


def info_model_command(args: argparse.Namespace) -> None:
    """Show detailed information about a model."""
    manager = ModelChecksumManager(args.models_dir)
    
    try:
        info = manager.get_model_info(args.model_name)
        
        if not info:
            logger.error(f"Model '{args.model_name}' not found")
            sys.exit(1)
        
        logger.info(f"Model: {info['model_name']}")
        logger.info(f"Created: {info['created_at']}")
        logger.info(f"Total files: {info['total_files']}")
        logger.info(f"Total size: {info['total_size_bytes'] / (1024 * 1024):.1f}MB")
        logger.info("Files:")
        
        for file_name, file_info in info["files"].items():
            size_mb = file_info["size_bytes"] / (1024 * 1024)
            logger.info(f"  {file_name}:")
            logger.info(f"    Checksum: {file_info['checksum']}")
            logger.info(f"    Size: {size_mb:.1f}MB")
            logger.info(f"    Modified: {file_info['last_modified']}")
            
    except ModelIntegrityError as e:
        logger.error(f"Failed to get model info: {e}")
        sys.exit(1)


def remove_model_command(args: argparse.Namespace) -> None:
    """Remove a model from the registry."""
    manager = ModelChecksumManager(args.models_dir)
    
    try:
        manager.remove_model(args.model_name)
        logger.info(f"✓ Model '{args.model_name}' removed from registry")
        
    except ModelIntegrityError as e:
        logger.error(f"Failed to remove model: {e}")
        sys.exit(1)


def verify_startup_command(args: argparse.Namespace) -> None:
    """Verify all critical models for application startup."""
    try:
        result = verify_startup_models()
        
        if result:
            logger.info("✓ All startup models verification PASSED")
        else:
            logger.error("✗ Startup models verification FAILED")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Startup verification failed: {e}")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Model Integrity Management for Clarity Loop Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register a model with auto-discovery
  python model_integrity_cli.py register pat_v1 --models-dir models/pat

  # Register a model with specific files
  python model_integrity_cli.py register gemini_v2 --models-dir models/gemini \\
    --files model.pth config.json

  # Verify a specific model
  python model_integrity_cli.py verify --models-dir models/pat --model pat_v1

  # Verify all models in a directory
  python model_integrity_cli.py verify --models-dir models/pat

  # List all registered models
  python model_integrity_cli.py list --models-dir models/pat

  # Show detailed model information
  python model_integrity_cli.py info pat_v1 --models-dir models/pat

  # Verify all critical models for startup
  python model_integrity_cli.py verify-startup
        """
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register command
    register_parser = subparsers.add_parser("register", help="Register a new model")
    register_parser.add_argument("model_name", help="Name of the model to register")
    register_parser.add_argument(
        "--models-dir", default="models",
        help="Directory containing model files (default: models)"
    )
    register_parser.add_argument(
        "--files", nargs="+",
        help="Specific model files to include (auto-discover if not specified)"
    )
    register_parser.set_defaults(func=register_model_command)

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify model integrity")
    verify_parser.add_argument(
        "--model", dest="model_name",
        help="Specific model to verify (verify all if not specified)"
    )
    verify_parser.add_argument(
        "--models-dir", default="models",
        help="Directory containing model files (default: models)"
    )
    verify_parser.set_defaults(func=verify_model_command)

    # List command
    list_parser = subparsers.add_parser("list", help="List registered models")
    list_parser.add_argument(
        "--models-dir", default="models",
        help="Directory containing model files (default: models)"
    )
    list_parser.set_defaults(func=list_models_command)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show detailed model information")
    info_parser.add_argument("model_name", help="Name of the model")
    info_parser.add_argument(
        "--models-dir", default="models",
        help="Directory containing model files (default: models)"
    )
    info_parser.set_defaults(func=info_model_command)

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove model from registry")
    remove_parser.add_argument("model_name", help="Name of the model to remove")
    remove_parser.add_argument(
        "--models-dir", default="models",
        help="Directory containing model files (default: models)"
    )
    remove_parser.set_defaults(func=remove_model_command)

    # Verify startup command
    startup_parser = subparsers.add_parser(
        "verify-startup", 
        help="Verify all critical models for application startup"
    )
    startup_parser.set_defaults(func=verify_startup_command)

    return parser


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main() 