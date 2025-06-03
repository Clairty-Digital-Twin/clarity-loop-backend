#!/usr/bin/env python3
"""Debug attention weight structures in PAT models."""

import logging

import h5py  # type: ignore[import-untyped]
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_DISPLAY_SIZE = 20
EXPECTED_DIMS = 3
EXPECTED_SIZE = 96


def analyze_attention_weights(h5_path: str, _model_name: str) -> None:
    """Analyze attention weight structure in detail."""
    with h5py.File(h5_path, 'r') as f:
        # Find all encoder layers
        encoder_layers = [k for k in f if 'encoder_layer' in k and 'transformer' in k]  # type: ignore[operator]

        for layer_name in encoder_layers:
            layer_group = f[layer_name]

            # Look for attention subgroup
            attention_keys = [k for k in layer_group if 'attention' in k]  # type: ignore[operator]

            for attn_key in attention_keys:
                attn_group = layer_group[attn_key]  # type: ignore[index]

                # Examine Q, K, V weights
                for qkv_name in ['query', 'key', 'value']:
                    if qkv_name in attn_group:  # type: ignore[operator]
                        qkv_group = attn_group[qkv_name]  # type: ignore[index]

                        for weight_key in qkv_group:  # type: ignore[misc]
                            if 'kernel' in weight_key or 'bias' in weight_key:
                                weight = qkv_group[weight_key]  # type: ignore[index]

                                # Show actual values for small arrays
                                if weight.size <= MAX_DISPLAY_SIZE:  # type: ignore[attr-defined]
                                    pass

                # Check attention output
                if 'attention_output' in attn_group:  # type: ignore[operator]
                    output_group = attn_group['attention_output']  # type: ignore[index]
                    for weight_key in output_group:  # type: ignore[misc]
                        if 'kernel' in weight_key or 'bias' in weight_key:
                            weight = output_group[weight_key]  # type: ignore[index]


def debug_specific_weights() -> None:
    """Debug specific weight issues we're seeing."""
    # Let's specifically debug the shape issue we're seeing

    with h5py.File("models/PAT-M_29k_weights.h5", 'r') as f:
        # Find the specific weight causing issues
        layer_group = f['encoder_layer_1_transformer']
        attn_group = layer_group['encoder_layer_1_attention']  # type: ignore[index]

        for qkv_name in ['query', 'key', 'value']:
            if qkv_name in attn_group:  # type: ignore[operator]
                qkv_group = attn_group[qkv_name]  # type: ignore[index]
                if 'kernel:0' in qkv_group:  # type: ignore[operator]
                    weight = qkv_group['kernel:0']  # type: ignore[index]

                    # Let's try to understand the dimensions
                    if len(weight.shape) == EXPECTED_DIMS:  # type: ignore[attr-defined]
                        _dim1, _dim2, _dim3 = weight.shape  # type: ignore[attr-defined]

                        # Try different reshaping strategies
                        flat_size = weight.size  # type: ignore[attr-defined]

                        # Option 1: Direct to (96, 96)
                        if flat_size == EXPECTED_SIZE * EXPECTED_SIZE:
                            pass

                        # Option 2: Permute then reshape
                        if len(weight.shape) == EXPECTED_DIMS:  # type: ignore[attr-defined]
                            weight_data = weight[:]  # type: ignore[index]
                            # Try different permutations
                            perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
                            for perm in perms:
                                try:
                                    permuted = np.transpose(weight_data, perm)  # type: ignore[arg-type]
                                    reshaped = permuted.reshape(EXPECTED_SIZE, -1)
                                    if reshaped.shape[1] == EXPECTED_SIZE:
                                        pass
                                except (ValueError, RuntimeError) as e:
                                    logger.debug("Reshape failed for permutation %s: %s", perm, e)
                                    continue


if __name__ == "__main__":
    analyze_attention_weights("models/PAT-M_29k_weights.h5", "PAT-M")
    debug_specific_weights()
