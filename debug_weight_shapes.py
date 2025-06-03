#!/usr/bin/env python3
"""Debug exact weight shapes in PAT-M model."""

import logging

import h5py  # type: ignore[import-untyped]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug_weight_shapes() -> None:
    """Debug exact weight shapes in PAT-M model."""
    h5_path = "models/PAT-M_29k_weights.h5"

    with h5py.File(h5_path, 'r') as f:
        logger.info("=== PAT-M Weight Shapes ===")

        # Dense layer
        if 'dense' in f and 'dense' in f['dense']:
            dense_group = f['dense']['dense']
            logger.info("Dense kernel: %s", dense_group['kernel:0'].shape)  # type: ignore[attr-defined]
            logger.info("Dense bias: %s", dense_group['bias:0'].shape)  # type: ignore[attr-defined]

        # First transformer layer attention
        if 'encoder_layer_1_transformer' in f:
            layer = f['encoder_layer_1_transformer']
            if 'encoder_layer_1_attention' in layer:
                attn = layer['encoder_layer_1_attention']

                # Q, K, V projections
                for qkv in ['query', 'key', 'value']:
                    if qkv in attn:
                        qkv_group = attn[qkv]
                        logger.info("Attention %s kernel: %s", qkv, qkv_group['kernel:0'].shape)  # type: ignore[attr-defined]
                        logger.info("Attention %s bias: %s", qkv, qkv_group['bias:0'].shape)  # type: ignore[attr-defined]

                # Output projection
                if 'attention_output' in attn:
                    out_group = attn['attention_output']
                    logger.info("Attention output kernel: %s", out_group['kernel:0'].shape)  # type: ignore[attr-defined]
                    logger.info("Attention output bias: %s", out_group['bias:0'].shape)  # type: ignore[attr-defined]

            # FF layers
            if 'encoder_layer_1_ff1' in layer:
                ff1_group = layer['encoder_layer_1_ff1']
                logger.info("FF1 kernel: %s", ff1_group['kernel:0'].shape)  # type: ignore[attr-defined]
                logger.info("FF1 bias: %s", ff1_group['bias:0'].shape)  # type: ignore[attr-defined]

            if 'encoder_layer_1_ff2' in layer:
                ff2_group = layer['encoder_layer_1_ff2']
                logger.info("FF2 kernel: %s", ff2_group['kernel:0'].shape)  # type: ignore[attr-defined]
                logger.info("FF2 bias: %s", ff2_group['bias:0'].shape)  # type: ignore[attr-defined]


if __name__ == "__main__":
    debug_weight_shapes()
