#!/usr/bin/env python3
"""Debug script to analyze attention weight shapes in PAT H5 files."""

import contextlib

import h5py
import numpy as np


def analyze_attention_weights(h5_path, model_name) -> None:
    """Analyze attention weight structure in detail."""
    with h5py.File(h5_path, 'r') as f:
        # Find all encoder layers
        encoder_layers = [k for k in f if 'encoder_layer' in k and 'transformer' in k]

        for layer_name in encoder_layers:
            layer_group = f[layer_name]

            # Look for attention subgroup
            attention_keys = [k for k in layer_group if 'attention' in k]

            for attn_key in attention_keys:
                attn_group = layer_group[attn_key]

                # Examine Q, K, V weights
                for qkv_name in ['query', 'key', 'value']:
                    if qkv_name in attn_group:
                        qkv_group = attn_group[qkv_name]

                        for weight_key in qkv_group:
                            if 'kernel' in weight_key or 'bias' in weight_key:
                                weight = qkv_group[weight_key]

                                # Show actual values for small arrays
                                if weight.size <= 20:
                                    pass

                # Check attention output
                if 'attention_output' in attn_group:
                    output_group = attn_group['attention_output']
                    for weight_key in output_group:
                        if 'kernel' in weight_key or 'bias' in weight_key:
                            weight = output_group[weight_key]


def main() -> None:
    """Main analysis function."""
    models = [
        ("models/PAT-S_29k_weights.h5", "PAT-S"),
        ("models/PAT-M_29k_weights.h5", "PAT-M"),
        ("models/PAT-L_29k_weights.h5", "PAT-L"),
    ]

    for h5_path, model_name in models:
        with contextlib.suppress(Exception):
            analyze_attention_weights(h5_path, model_name)

    # Let's specifically debug the shape issue we're seeing

    with h5py.File("models/PAT-M_29k_weights.h5", 'r') as f:
        # Find the specific weight causing issues
        layer_group = f['encoder_layer_1_transformer']
        attn_group = layer_group['encoder_layer_1_attention']

        for qkv_name in ['query', 'key', 'value']:
            if qkv_name in attn_group:
                qkv_group = attn_group[qkv_name]
                if 'kernel:0' in qkv_group:
                    weight = qkv_group['kernel:0']

                    # Let's try to understand the dimensions
                    if len(weight.shape) == 3:
                        _dim1, _dim2, _dim3 = weight.shape

                        # Try different reshaping strategies
                        flat_size = weight.size

                        # Option 1: Direct to (96, 96)
                        if flat_size == 96 * 96:
                            pass

                        # Option 2: Permute then reshape
                        if len(weight.shape) == 3:
                            weight_data = weight[:]
                            # Try different permutations
                            perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
                            for perm in perms:
                                try:
                                    permuted = np.transpose(weight_data, perm)
                                    reshaped = permuted.reshape(96, -1)
                                    if reshaped.shape[1] == 96:
                                        pass
                                except:
                                    continue


if __name__ == "__main__":
    main()
