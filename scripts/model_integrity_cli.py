#!/usr/bin/env python3
"""Model Integrity CLI Tool.

Command-line interface for managing ML model integrity verification.
Provides commands to register, verify, list, and manage model checksums.
"""

import argparse
import logging
from pathlib import Path
import sys

from clarity.ml.model_integrity import (
    ModelChecksumManager,
    ModelIntegrityError,
    verify_startup_models,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def register_command(args: argparse.Namespace) -> None:
    """Register a new model with the integrity manager."""
    try:
        manager = ModelChecksumManager(args.models_dir)

        # Get model files from directory
        models_dir = Path(args.models_dir)
        if args.model_files:
            model_files = [str(Path(f)) for f in args.model_files]
        else:
            # Auto-discover model files in the model directory
            model_dir = models_dir / args.model_name
            if not model_dir.exists():
                logger.error("Model directory %s does not exist", model_dir)
                return

            model_files_paths = list(model_dir.glob("**/*"))
            model_files = [str(f) for f in model_files_paths if f.is_file()]

            if not model_files:
                logger.error("No model files found in %s", models_dir)
                return

        logger.info(
            "Registering model '%s' with files: %s", args.model_name, model_files
        )
        manager.register_model(args.model_name, model_files)
        logger.info("✓ Model registered successfully")

    except ModelIntegrityError:
        logger.exception("Failed to register model")
        sys.exit(1)


def verify_command(args: argparse.Namespace) -> None:
    """Verify model integrity."""
    try:
        manager = ModelChecksumManager(args.models_dir)

        if args.model_name:
            # Verify specific model
            result = manager.verify_model_integrity(args.model_name)
            if result:
                logger.info("✓ Model '%s' verification PASSED", args.model_name)
            else:
                logger.error("✗ Model '%s' verification FAILED", args.model_name)
                sys.exit(1)
        else:
            # Verify all models
            results = manager.verify_all_models()
            passed_count = sum(results.values())
            total_count = len(results)

            logger.info("Verification Results:")
            for model_name, passed in results.items():
                status = "✓ PASSED" if passed else "✗ FAILED"
                logger.info("%s: %s", model_name, status)

            if passed_count == total_count:
                logger.info("✓ All models (%d) verification PASSED", total_count)
            else:
                logger.error(
                    "✗ Verification: %d/%d models passed", passed_count, total_count
                )
                sys.exit(1)

    except ModelIntegrityError:
        logger.exception("Verification failed")
        sys.exit(1)


def list_command(args: argparse.Namespace) -> None:
    """List registered models."""
    try:
        manager = ModelChecksumManager(args.models_dir)
        models = manager.list_registered_models()

        if not models:
            logger.info("No models registered in %s", args.models_dir)
            return

        logger.info("Registered models in %s:", args.models_dir)
        for model_name in models:
            info = manager.get_model_info(model_name)
            if info:
                file_count = info.get("total_files", 0)
                size_bytes = info.get("total_size_bytes", 0)
                size_mb = size_bytes / (1024 * 1024)
                created = info.get("created_at", "unknown")
                logger.info(
                    "  %s: %d files, %.1fMB, created %s",
                    model_name,
                    file_count,
                    size_mb,
                    created,
                )
            else:
                logger.info("  %s: (info unavailable)", model_name)

    except ModelIntegrityError:
        logger.exception("Failed to list models")
        sys.exit(1)


def info_command(args: argparse.Namespace) -> None:
    """Show detailed information about a specific model."""
    try:
        manager = ModelChecksumManager(args.models_dir)
        info = manager.get_model_info(args.model_name)

        if not info:
            logger.error("Model '%s' not found", args.model_name)
            sys.exit(1)

        logger.info("Model: %s", info["model_name"])
        logger.info("Created: %s", info["created_at"])
        logger.info("Total files: %d", info["total_files"])
        logger.info("Total size: %.1fMB", info["total_size_bytes"] / (1024 * 1024))
        logger.info("Files:")

        for file_name, file_info in info["files"].items():
            size_mb = file_info["size_bytes"] / (1024 * 1024)
            logger.info("  %s:", file_name)
            logger.info("    Checksum: %s", file_info["checksum"])
            logger.info("    Size: %.1fMB", size_mb)
            logger.info("    Modified: %s", file_info["last_modified"])

    except ModelIntegrityError:
        logger.exception("Failed to get model info")
        sys.exit(1)


def remove_command(args: argparse.Namespace) -> None:
    """Remove a model from the registry."""
    try:
        manager = ModelChecksumManager(args.models_dir)
        manager.remove_model(args.model_name)
        logger.info("✓ Model '%s' removed from registry", args.model_name)

    except ModelIntegrityError:
        logger.exception("Failed to remove model")
        sys.exit(1)


def verify_startup_command(_args: argparse.Namespace) -> None:
    """Verify all critical models for application startup."""
    try:
        logger.info("Verifying critical models for startup...")
        success = verify_startup_models()

        if success:
            logger.info("✓ All critical models verified successfully")
        else:
            logger.error("✗ Critical model verification failed")
            sys.exit(1)

    except Exception:
        logger.exception("Startup verification failed")
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
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register command
    register_parser = subparsers.add_parser("register", help="Register a new model")
    register_parser.add_argument("model_name", help="Name of the model to register")
    register_parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing model files (default: models)",
    )
    register_parser.add_argument(
        "--files",
        nargs="+",
        help="Specific model files to include (auto-discover if not specified)",
    )
    register_parser.set_defaults(func=register_command)

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify model integrity")
    verify_parser.add_argument(
        "--model",
        dest="model_name",
        help="Specific model to verify (verify all if not specified)",
    )
    verify_parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing model files (default: models)",
    )
    verify_parser.set_defaults(func=verify_command)

    # List command
    list_parser = subparsers.add_parser("list", help="List registered models")
    list_parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing model files (default: models)",
    )
    list_parser.set_defaults(func=list_command)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show detailed model information")
    info_parser.add_argument("model_name", help="Name of the model")
    info_parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing model files (default: models)",
    )
    info_parser.set_defaults(func=info_command)

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove model from registry")
    remove_parser.add_argument("model_name", help="Name of the model to remove")
    remove_parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing model files (default: models)",
    )
    remove_parser.set_defaults(func=remove_command)

    # Verify startup command
    startup_parser = subparsers.add_parser(
        "verify-startup", help="Verify all critical models for application startup"
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
    except Exception:
        logger.exception("Unexpected error occurred")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
