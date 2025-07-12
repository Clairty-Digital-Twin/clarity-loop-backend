#!/bin/bash
# Create symlinks for backward compatibility with old model names
set -e

MODEL_DIR="${MODEL_DIR:-/app/models/pat}"
cd "$MODEL_DIR"

# Create symlinks if actual files exist
if [ -f "PAT-S_29k_weight_transformer.h5" ]; then
    ln -sf "PAT-S_29k_weight_transformer.h5" "PAT-S_29k_weights.h5"
    echo "✅ Created symlink for PAT-S model"
fi

if [ -f "PAT-M_29k_weight_transformer.h5" ]; then
    ln -sf "PAT-M_29k_weight_transformer.h5" "PAT-M_29k_weights.h5"
    echo "✅ Created symlink for PAT-M model"
fi

if [ -f "PAT-L_91k_weight_transformer.h5" ]; then
    ln -sf "PAT-L_91k_weight_transformer.h5" "PAT-L_29k_weights.h5"
    echo "✅ Created symlink for PAT-L model"
fi

# Also handle the /models/pat directory if it exists
if [ -d "/models/pat" ]; then
    cd /models/pat
    
    if [ -f "PAT-S_29k_weight_transformer.h5" ]; then
        ln -sf "PAT-S_29k_weight_transformer.h5" "PAT-S_29k_weights.h5"
    fi
    
    if [ -f "PAT-M_29k_weight_transformer.h5" ]; then
        ln -sf "PAT-M_29k_weight_transformer.h5" "PAT-M_29k_weights.h5"
    fi
    
    if [ -f "PAT-L_91k_weight_transformer.h5" ]; then
        ln -sf "PAT-L_91k_weight_transformer.h5" "PAT-L_29k_weights.h5"
    fi
fi
