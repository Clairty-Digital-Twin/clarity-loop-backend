#!/usr/bin/env python3
"""Debug attention weight structures in PAT models."""

import logging
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_DISPLAY_SIZE = 20
EXPECTED_DIMS = 3
EXPECTED_SIZE = 96


def _process_qkv_weights(qkv_group: Any, qkv_name: str) -> None:
    """Process query, key, or value weights from an attention group."""
    for weight_key in qkv_group:  # type: ignore[misc]
        if 'kernel' in weight_key or 'bias' in weight_key:
            weight = qkv_group[weight_key]  # type: ignore[index]
            # Show actual values for small arrays
            if weight.size <= MAX_DISPLAY_SIZE:  # type: ignore[attr-defined]
                logger.debug("Small weight array found: %s.%s", qkv_name, weight_key)


def _process_attention_output(attn_group: Any) -> None:
    """Process attention output weights."""
    if 'attention_output' not in attn_group:  # type: ignore[operator]
        return

    output_group = attn_group['attention_output']  # type: ignore[index]
    for weight_key in output_group:  # type: ignore[misc]
        if 'kernel' in weight_key or 'bias' in weight_key:
            weight = output_group[weight_key]  # type: ignore[index]
            logger.debug("Attention output weight: %s", weight_key)


def _process_attention_group(attn_group: Any, attn_key: str) -> None:
    """Process a single attention group."""
    # Examine Q, K, V weights
    for qkv_name in ['query', 'key', 'value']:
        if qkv_name in attn_group:  # type: ignore[operator]
            qkv_group = attn_group[qkv_name]  # type: ignore[index]
            _process_qkv_weights(qkv_group, qkv_name)

    # Check attention output
    _process_attention_output(attn_group)


def _process_layer(layer_group: Any, layer_name: str) -> None:
    """Process a single transformer layer."""
    # Look for attention subgroup
    attention_keys = [k for k in layer_group if 'attention' in k]  # type: ignore[operator]

    for attn_key in attention_keys:
        attn_group = layer_group[attn_key]  # type: ignore[index]
        _process_attention_group(attn_group, attn_key)


def analyze_attention_weights(h5_path: str, _model_name: str) -> None:
    """Analyze attention weight structure in detail."""
    with h5py.File(h5_path, 'r') as f:
        # Find all encoder layers
        encoder_layers = [k for k in f if 'encoder_layer' in k and 'transformer' in k]  # type: ignore[operator]

        for layer_name in encoder_layers:
            if layer_name is not None:
                layer_group = f[layer_name]
                _process_layer(layer_group, layer_name)


def _try_reshape_permutations(weight_data: Any) -> None:
    """Try different permutations to reshape weight data."""
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    for perm in perms:
        try:
            permuted = np.transpose(weight_data, perm)  # type: ignore[arg-type]
            reshaped = permuted.reshape(EXPECTED_SIZE, -1)
            if reshaped.shape[1] == EXPECTED_SIZE:
                logger.debug("Successfully reshaped with permutation %s", perm)
                return
        except (ValueError, RuntimeError) as e:
            logger.debug("Reshape failed for permutation %s: %s", perm, e)


def _analyze_weight_dimensions(weight: Any, qkv_name: str) -> None:
    """Analyze weight dimensions and attempt reshaping."""
    # Let's try to understand the dimensions
    if len(weight.shape) != EXPECTED_DIMS:  # type: ignore[attr-defined]
        return

    _dim1, _dim2, _dim3 = weight.shape  # type: ignore[attr-defined]
    flat_size = weight.size  # type: ignore[attr-defined]

    # Option 1: Direct to (96, 96)
    if flat_size == EXPECTED_SIZE * EXPECTED_SIZE:
        logger.debug("Weight %s can be directly reshaped to (%d, %d)",
                    qkv_name, EXPECTED_SIZE, EXPECTED_SIZE)
        return

    # Option 2: Permute then reshape
    weight_data = weight[:]  # type: ignore[index]
    _try_reshape_permutations(weight_data)


def _process_qkv_kernel(qkv_group: Any, qkv_name: str) -> None:
    """Process kernel weights for query, key, or value."""
    if 'kernel:0' not in qkv_group:  # type: ignore[operator]
        return

    weight = qkv_group['kernel:0']  # type: ignore[index]
    _analyze_weight_dimensions(weight, qkv_name)


def debug_specific_weights() -> None:
    """Debug specific weight issues we're seeing."""
    try:
        with h5py.File("models/PAT-M_29k_weights.h5", 'r') as f:
            # Find the specific weight causing issues
            layer_group = f['encoder_layer_1_transformer']
            attn_group = layer_group['encoder_layer_1_attention']  # type: ignore[index]

            for qkv_name in ['query', 'key', 'value']:
                if qkv_name in attn_group:  # type: ignore[operator]
                    qkv_group = attn_group[qkv_name]  # type: ignore[index]
                    _process_qkv_kernel(qkv_group, qkv_name)
    except (OSError, KeyError) as e:
        logger.error("Failed to debug specific weights: %s", e)


if __name__ == "__main__":
    analyze_attention_weights("models/PAT-M_29k_weights.h5", "PAT-M")
    debug_specific_weights()
