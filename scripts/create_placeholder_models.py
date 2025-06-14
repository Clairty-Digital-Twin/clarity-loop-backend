#!/usr/bin/env python3
"""Create placeholder PAT model files for testing.

This script creates minimal HDF5 files that match the structure expected
by the PAT model loader but contain random weights for testing purposes.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

try:
    import h5py
except ImportError:
    print("Error: h5py is required. Install with: pip install h5py")
    sys.exit(1)

# Model configurations matching PAT architecture
MODEL_CONFIGS = {
    "PAT-S_29k_weights.h5": {
        "num_layers": 1,
        "num_heads": 6,
        "embed_dim": 96,
        "ff_dim": 256,
        "patch_size": 18,
    },
    "PAT-M_29k_weights.h5": {
        "num_layers": 2,
        "num_heads": 12,
        "embed_dim": 96,
        "ff_dim": 256,
        "patch_size": 18,
    },
    "PAT-L_29k_weights.h5": {
        "num_layers": 4,
        "num_heads": 12,
        "embed_dim": 96,
        "ff_dim": 256,
        "patch_size": 9,
    },
}


def create_placeholder_weights(config: dict[str, int]) -> dict[str, np.ndarray]:
    """Create placeholder weights matching PAT architecture."""
    weights = {}
    rng = np.random.default_rng(42)  # Use new random generator with seed

    # Patch embedding layer
    weights["patch_embedding/kernel:0"] = rng.standard_normal(
        (
            config["patch_size"],
            config["embed_dim"],
        )
    ).astype(np.float32)
    weights["patch_embedding/bias:0"] = rng.standard_normal(config["embed_dim"]).astype(
        np.float32
    )

    # Transformer layers
    for layer_idx in range(config["num_layers"]):
        prefix = f"transformer_block_{layer_idx}"

        # Multi-head attention
        weights[f"{prefix}/multi_head_attention/query/kernel:0"] = rng.standard_normal(
            (
                config["embed_dim"],
                config["embed_dim"],
            )
        ).astype(np.float32)
        weights[f"{prefix}/multi_head_attention/key/kernel:0"] = rng.standard_normal(
            (
                config["embed_dim"],
                config["embed_dim"],
            )
        ).astype(np.float32)
        weights[f"{prefix}/multi_head_attention/value/kernel:0"] = rng.standard_normal(
            (
                config["embed_dim"],
                config["embed_dim"],
            )
        ).astype(np.float32)
        weights[f"{prefix}/multi_head_attention/attention_output/kernel:0"] = (
            rng.standard_normal((config["embed_dim"], config["embed_dim"])).astype(
                np.float32
            )
        )

        # Layer normalization
        weights[f"{prefix}/layer_normalization/gamma:0"] = np.ones(
            config["embed_dim"]
        ).astype(np.float32)
        weights[f"{prefix}/layer_normalization/beta:0"] = np.zeros(
            config["embed_dim"]
        ).astype(np.float32)
        weights[f"{prefix}/layer_normalization_1/gamma:0"] = np.ones(
            config["embed_dim"]
        ).astype(np.float32)
        weights[f"{prefix}/layer_normalization_1/beta:0"] = np.zeros(
            config["embed_dim"]
        ).astype(np.float32)

        # Feed-forward network
        weights[f"{prefix}/ffn/dense_1/kernel:0"] = rng.standard_normal(
            (
                config["embed_dim"],
                config["ff_dim"],
            )
        ).astype(np.float32)
        weights[f"{prefix}/ffn/dense_1/bias:0"] = rng.standard_normal(
            config["ff_dim"]
        ).astype(np.float32)
        weights[f"{prefix}/ffn/dense_2/kernel:0"] = rng.standard_normal(
            (
                config["ff_dim"],
                config["embed_dim"],
            )
        ).astype(np.float32)
        weights[f"{prefix}/ffn/dense_2/bias:0"] = rng.standard_normal(
            config["embed_dim"]
        ).astype(np.float32)

    # Output layers
    weights["classifier/kernel:0"] = rng.standard_normal(
        (config["embed_dim"], 18)  # 18 output classes
    ).astype(np.float32)
    weights["classifier/bias:0"] = rng.standard_normal(18).astype(np.float32)

    return weights


def create_model_file(filename: str, config: dict[str, int], output_dir: Path) -> None:
    """Create a placeholder HDF5 model file."""
    filepath = output_dir / filename

    # Create placeholder weights
    weights = create_placeholder_weights(config)

    # Save to HDF5
    with h5py.File(filepath, "w") as f:
        # Create model weights group
        model_weights = f.create_group("model_weights")

        # Add each weight array
        for name, array in weights.items():
            model_weights.create_dataset(name, data=array)

        # Add metadata
        f.attrs["keras_version"] = "2.13.0"
        f.attrs["backend"] = "tensorflow"
        f.attrs["model_config"] = json.dumps(
            {
                "class_name": "PATModel",
                "config": config,
            }
        )

    # Calculate SHA256 checksum
    sha256_hash = hashlib.sha256()
    with filepath.open("rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    checksum = sha256_hash.hexdigest()

    return filepath, checksum


def main() -> None:
    """Create placeholder model files."""
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / "models" / "pat"

    # Create models directory
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Creating placeholder PAT model files...")
    print(f"Output directory: {models_dir}")
    print()

    checksums = {}

    for filename, config in MODEL_CONFIGS.items():
        print(f"Creating {filename}...")
        filepath, checksum = create_model_file(filename, config, models_dir)
        checksums[filename] = checksum
        print(f"  ✓ Created: {filepath}")
        print(f"  ✓ SHA256: {checksum}")
        print()

    # Save checksums to file
    checksum_file = models_dir / "checksums.json"
    with checksum_file.open("w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2)
    print(f"Saved checksums to: {checksum_file}")

    # Print environment variables for deployment
    print("\nEnvironment variables for deployment:")
    print(f'export PAT_S_CHECKSUM="{checksums["PAT-S_29k_weights.h5"]}"')
    print(f'export PAT_M_CHECKSUM="{checksums["PAT-M_29k_weights.h5"]}"')
    print(f'export PAT_L_CHECKSUM="{checksums["PAT-L_29k_weights.h5"]}"')


if __name__ == "__main__":
    main()
