#!/usr/bin/env python3
"""Debug weight shapes for proper conversion."""

import h5py


def debug_weight_shapes() -> None:
    """Debug exact weight shapes in PAT-M model."""
    h5_path = "models/PAT-M_29k_weights.h5"

    with h5py.File(h5_path, 'r') as f:

        # Dense layer
        if 'dense' in f and 'dense' in f['dense']:
            dense_group = f['dense']['dense']

        # First transformer layer attention
        layer_key = 'encoder_layer_1_transformer'
        if layer_key in f:
            layer = f[layer_key]

            # Attention layer
            if 'encoder_layer_1_attention' in layer:
                attn = layer['encoder_layer_1_attention']

                # QKV weights
                for qkv in ['query', 'key', 'value']:
                    if qkv in attn:
                        qkv_group = attn[qkv]

                # Output projection
                if 'attention_output' in attn:
                    out_group = attn['attention_output']

            # FF layers
            if 'encoder_layer_1_ff1' in layer:
                ff1_group = layer['encoder_layer_1_ff1']

            if 'encoder_layer_1_ff2' in layer:
                ff2_group = layer['encoder_layer_1_ff2']


if __name__ == "__main__":
    debug_weight_shapes()
