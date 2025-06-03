#!/usr/bin/env python3
"""Debug script to analyze attention weight shapes in PAT H5 files."""

import h5py
import numpy as np

def analyze_attention_weights(h5_path, model_name):
    """Analyze attention weight structure in detail."""
    print(f"\n=== {model_name} Attention Weights Analysis ===")
    
    with h5py.File(h5_path, 'r') as f:
        # Find all encoder layers
        encoder_layers = [k for k in f.keys() if 'encoder_layer' in k and 'transformer' in k]
        print(f"Found encoder layers: {encoder_layers}")
        
        for layer_name in encoder_layers:
            print(f"\n--- {layer_name} ---")
            layer_group = f[layer_name]
            
            # Look for attention subgroup
            attention_keys = [k for k in layer_group.keys() if 'attention' in k]
            print(f"Attention keys: {attention_keys}")
            
            for attn_key in attention_keys:
                print(f"\n  {attn_key}:")
                attn_group = layer_group[attn_key]
                
                # Examine Q, K, V weights
                for qkv_name in ['query', 'key', 'value']:
                    if qkv_name in attn_group:
                        print(f"    {qkv_name}:")
                        qkv_group = attn_group[qkv_name]
                        
                        for weight_key in qkv_group.keys():
                            if 'kernel' in weight_key or 'bias' in weight_key:
                                weight = qkv_group[weight_key]
                                print(f"      {weight_key}: shape={weight.shape}, size={weight.size}")
                                
                                # Show actual values for small arrays
                                if weight.size <= 20:
                                    print(f"        Values: {weight[:]}")
                
                # Check attention output
                if 'attention_output' in attn_group:
                    print("    attention_output:")
                    output_group = attn_group['attention_output']
                    for weight_key in output_group.keys():
                        if 'kernel' in weight_key or 'bias' in weight_key:
                            weight = output_group[weight_key]
                            print(f"      {weight_key}: shape={weight.shape}, size={weight.size}")

def main():
    """Main analysis function."""
    models = [
        ("models/PAT-S_29k_weights.h5", "PAT-S"),
        ("models/PAT-M_29k_weights.h5", "PAT-M"),
        ("models/PAT-L_29k_weights.h5", "PAT-L"),
    ]
    
    for h5_path, model_name in models:
        try:
            analyze_attention_weights(h5_path, model_name)
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
    
    # Let's specifically debug the shape issue we're seeing
    print("\n=== DEBUGGING SHAPE MISMATCH ===")
    
    with h5py.File("models/PAT-M_29k_weights.h5", 'r') as f:
        # Find the specific weight causing issues
        layer_group = f['encoder_layer_1_transformer']
        attn_group = layer_group['encoder_layer_1_attention']
        
        for qkv_name in ['query', 'key', 'value']:
            if qkv_name in attn_group:
                qkv_group = attn_group[qkv_name]
                if 'kernel:0' in qkv_group:
                    weight = qkv_group['kernel:0']
                    print(f"{qkv_name} kernel shape: {weight.shape}")
                    print(f"  Total elements: {weight.size}")
                    print(f"  Expected PyTorch shape: (96, 96) = {96*96} elements")
                    
                    # Let's try to understand the dimensions
                    if len(weight.shape) == 3:
                        dim1, dim2, dim3 = weight.shape
                        print(f"  TF dimensions: {dim1} x {dim2} x {dim3}")
                        print(f"  Possible interpretations:")
                        print(f"    - Input dim: {dim1}")
                        print(f"    - Num heads: {dim2}")
                        print(f"    - Head dim: {dim3}")
                        print(f"    - Head dim * num heads = {dim2 * dim3}")
                        
                        # Try different reshaping strategies
                        flat_size = weight.size
                        print(f"  Reshaping options for size {flat_size}:")
                        
                        # Option 1: Direct to (96, 96)
                        if flat_size == 96 * 96:
                            print("    ✅ Can reshape to (96, 96)")
                        else:
                            print(f"    ❌ Cannot reshape to (96, 96) - size mismatch")
                        
                        # Option 2: Permute then reshape
                        if len(weight.shape) == 3:
                            weight_data = weight[:]
                            # Try different permutations
                            perms = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
                            for perm in perms:
                                try:
                                    permuted = np.transpose(weight_data, perm)
                                    reshaped = permuted.reshape(96, -1)
                                    if reshaped.shape[1] == 96:
                                        print(f"    ✅ Permutation {perm} gives shape {reshaped.shape}")
                                except:
                                    continue

if __name__ == "__main__":
    main() 