#!/usr/bin/env python3
"""Debug script to explore H5 file structure for PAT models."""

import logging
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def explore_h5_structure(file_path: str, max_depth: int = 2) -> None:
    """Explore the structure of an H5 file."""
    logger.info("=== H5 File Structure: %s ===", file_path)

    def print_structure(name: str, obj: Any, depth: int = 0) -> None:
        indent = "  " * depth
        if isinstance(obj, h5py.Dataset):
            logger.info("%s%s: Dataset %s %s", indent, name, obj.shape, obj.dtype)  # type: ignore[attr-defined]
        elif isinstance(obj, h5py.Group) and depth < max_depth:
            logger.info("%s%s: Group", indent, name)
            for key in obj:
                if key is not None:
                    print_structure(key, obj[key], depth + 1)

    with h5py.File(file_path, 'r') as f:
        logger.info("Root keys: %s", list(f.keys()))
        for key in f:
            if key is not None:
                print_structure(key, f[key])


def analyze_weight_structure(file_path: str) -> None:
    """Analyze the weight structure specifically for PAT models."""
    logger.info("=== Weight Structure Analysis: %s ===", file_path)

    with h5py.File(file_path, 'r') as f:
        # Check for top level model weights
        if 'top_level_model_weights' in f:
            tlmw = f['top_level_model_weights']
            if isinstance(tlmw, h5py.Group):
                logger.info("TLMW keys: %s", list(tlmw.keys()))

        # Check input layer
        if 'inputs' in f:
            inputs = f['inputs']
            if isinstance(inputs, h5py.Group):
                logger.info("Inputs group keys: %s", list(inputs.keys()))
                for key in inputs:
                    if key is not None and isinstance(inputs[key], h5py.Dataset):
                        dataset = inputs[key]
                        logger.info("  %s: %s %s", key, dataset.shape, dataset.dtype)  # type: ignore[attr-defined]

        # Check transformer layers
        transformer_layers = [k for k in f if k is not None and 'encoder_layer' in k]
        logger.info("Found %d transformer layers", len(transformer_layers))

        if transformer_layers:
            layer = f[transformer_layers[0]]
            if isinstance(layer, h5py.Group):
                logger.info("First layer (%s) keys: %s", transformer_layers[0], list(layer.keys()))

        # Check output layers
        if 'dense' in f:
            dense = f['dense']
            if isinstance(dense, h5py.Group):
                logger.info("Dense layer keys: %s", list(dense.keys()))
                for key in dense:
                    if key is not None and isinstance(dense[key], h5py.Dataset):
                        dataset = dense[key]
                        logger.info("  %s: %s %s", key, dataset.shape, dataset.dtype)  # type: ignore[attr-defined]


def main() -> None:
    """Main function to analyze all PAT model files."""
    model_files = [
        "models/PAT-S_sleep_classification.h5",
        "models/PAT-M_mental_health.h5",
        "models/PAT-M_29k_weights.h5",
        "models/PAT-L_circadian_rhythm.h5"
    ]

    for model_file in model_files:
        if Path(model_file).exists():
            try:
                explore_h5_structure(model_file, max_depth=3)
                analyze_weight_structure(model_file)
            except (OSError, KeyError, ValueError):
                logger.exception("Error analyzing %s", model_file)
        else:
            logger.warning("File not found: %s", model_file)


if __name__ == "__main__":
    main()
