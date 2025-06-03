#!/usr/bin/env python3
"""Debug script to explore H5 file structure for PAT weights."""

from pathlib import Path

import h5py


def explore_h5_structure(filepath, max_depth=4):
    """Recursively explore H5 file structure."""
    print(f"\n=== Exploring {filepath} ===")

    with h5py.File(filepath, 'r') as f:
        def print_structure(name, obj, depth=0):
            indent = "  " * depth
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{name}: Dataset {obj.shape} {obj.dtype}")
                # Show first few values for small datasets
                if hasattr(obj, 'size') and obj.size < 20:
                    print(f"{indent}  Values: {obj[:]}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}{name}: Group")
                if depth < max_depth:
                    for key in obj.keys():
                        print_structure(f"{name}/{key}", obj[key], depth + 1)

        print("Root keys:", list(f.keys()))
        for key in f.keys():
            print_structure(key, f[key])


def analyze_weight_structure(filepath):
    """Analyze the weight structure to understand mapping."""
    print(f"\n=== Weight Analysis for {filepath} ===")

    with h5py.File(filepath, 'r') as f:
        # Check for common TensorFlow/Keras patterns
        if 'top_level_model_weights' in f:
            print("Found top_level_model_weights group")
            tlmw = f['top_level_model_weights']
            if isinstance(tlmw, h5py.Group):
                print("TLMW keys:", list(tlmw.keys()))

        # Check input layer
        if 'inputs' in f:
            inputs = f['inputs']
            if isinstance(inputs, h5py.Group):
                print("Inputs group keys:", list(inputs.keys()))
                for key in inputs.keys():
                    if isinstance(inputs[key], h5py.Dataset):
                        print(f"  {key}: {inputs[key].shape} {inputs[key].dtype}")

        # Check transformer layers
        transformer_layers = [k for k in f.keys() if 'encoder_layer' in k]
        print(f"Found {len(transformer_layers)} transformer layers")

        if transformer_layers:
            layer = f[transformer_layers[0]]
            if isinstance(layer, h5py.Group):
                print(f"First layer ({transformer_layers[0]}) keys:", list(layer.keys()))

        # Check output layers
        if 'dense' in f:
            dense = f['dense']
            if isinstance(dense, h5py.Group):
                print("Dense layer keys:", list(dense.keys()))
                for key in dense.keys():
                    if isinstance(dense[key], h5py.Dataset):
                        print(f"  {key}: {dense[key].shape} {dense[key].dtype}")


def main():
    """Main function to analyze all PAT model files."""
    model_files = [
        "models/PAT-S_29k_weights.h5",
        "models/PAT-M_29k_weights.h5",
        "models/PAT-L_29k_weights.h5"
    ]

    for model_file in model_files:
        if Path(model_file).exists():
            try:
                explore_h5_structure(model_file, max_depth=3)
                analyze_weight_structure(model_file)
            except Exception as e:
                print(f"Error analyzing {model_file}: {e}")
        else:
            print(f"File not found: {model_file}")


if __name__ == "__main__":
    main()
