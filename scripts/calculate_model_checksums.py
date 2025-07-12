#!/usr/bin/env python3
"""Calculate the correct checksums for PAT model files."""

import hashlib
import hmac
from pathlib import Path

MODEL_SIGNATURE_KEY = "pat_model_integrity_key_2025"


def calculate_file_checksum(filepath: str) -> str:
    """Calculate SHA-256 checksum with HMAC signature for file integrity."""
    try:
        sha256_hash = hashlib.sha256()

        # Read file in chunks to handle large files
        with Path(filepath).open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)

        # Create HMAC signature for additional security
        file_digest = sha256_hash.hexdigest()
        return hmac.new(
            MODEL_SIGNATURE_KEY.encode("utf-8"),
            file_digest.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    except OSError as e:
        print(f"Error calculating checksum for {filepath}: {e}")
        return ""


# Calculate checksums for all PAT models
models_dir = Path("models/pat")
for model_file in models_dir.glob("*.h5"):
    checksum = calculate_file_checksum(model_file)
    model_size = (
        "small"
        if "PAT-S" in model_file.name
        else ("medium" if "PAT-M" in model_file.name else "large")
    )
    print(
        f'    "{model_size}": "{checksum}",  # SHA-256 of authentic {model_file.name}'
    )
