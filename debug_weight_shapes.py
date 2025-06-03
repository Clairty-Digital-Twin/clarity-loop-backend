#!/usr/bin/env python3
"""Debug weight shapes for proper conversion."""

import h5py


def debug_weight_shapes():
    """Debug exact weight shapes in PAT-M model."""
    h5_path = "models/PAT-M_29k_weights.h5"

    with h5py.File(h5_path, 'r') as f:
        print("=== PAT-M Weight Shapes ===")

        # Dense layer
        if 'dense' in f and 'dense' in f['dense']:
            dense_group = f['dense']['dense']
            print(f"Dense kernel: {dense_group['kernel:0'].shape}")
            print(f"Dense bias: {dense_group['bias:0'].shape}")

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
                        print(f"Attention {qkv} kernel: {qkv_group['kernel:0'].shape}")
                        print(f"Attention {qkv} bias: {qkv_group['bias:0'].shape}")

                # Output projection
                if 'attention_output' in attn:
                    out_group = attn['attention_output']
                    print(f"Attention output kernel: {out_group['kernel:0'].shape}")
                    print(f"Attention output bias: {out_group['bias:0'].shape}")

            # FF layers
            if 'encoder_layer_1_ff1' in layer:
                ff1_group = layer['encoder_layer_1_ff1']
                print(f"FF1 kernel: {ff1_group['kernel:0'].shape}")
                print(f"FF1 bias: {ff1_group['bias:0'].shape}")

            if 'encoder_layer_1_ff2' in layer:
                ff2_group = layer['encoder_layer_1_ff2']
                print(f"FF2 kernel: {ff2_group['kernel:0'].shape}")
                print(f"FF2 bias: {ff2_group['bias:0'].shape}")


if __name__ == "__main__":
    debug_weight_shapes()
