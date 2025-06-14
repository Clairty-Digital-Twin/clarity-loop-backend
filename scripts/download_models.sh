#!/bin/bash
# Download ML models from S3 with checksum verification
set -e

echo "Starting model download process..."

# Model download directory
MODEL_DIR="${MODEL_DIR:-/app/models/pat}"
mkdir -p "$MODEL_DIR"

# S3 bucket configuration
S3_BUCKET="${S3_BUCKET:-s3://clarity-ml-models-124355672559}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Model configurations from environment variables
declare -A MODELS
MODELS["PAT-S"]="${PAT_S_MODEL_PATH:-${S3_BUCKET}/pat/PAT-S_29k_weights.h5}"
MODELS["PAT-M"]="${PAT_M_MODEL_PATH:-${S3_BUCKET}/pat/PAT-M_29k_weights.h5}"
MODELS["PAT-L"]="${PAT_L_MODEL_PATH:-${S3_BUCKET}/pat/PAT-L_29k_weights.h5}"

# Checksums from environment variables
declare -A CHECKSUMS
CHECKSUMS["PAT-S"]="${PAT_S_CHECKSUM}"
CHECKSUMS["PAT-M"]="${PAT_M_CHECKSUM}"
CHECKSUMS["PAT-L"]="${PAT_L_CHECKSUM}"

# Function to download and verify a model
download_and_verify_model() {
    local model_name=$1
    local s3_path=$2
    local expected_checksum=$3
    local filename=$(basename "$s3_path")
    local local_path="${MODEL_DIR}/${filename}"
    
    echo "Downloading ${model_name} from ${s3_path}..."
    
    # Download model from S3
    if aws s3 cp "$s3_path" "$local_path" --region "$AWS_REGION"; then
        echo "✓ Downloaded ${model_name} successfully"
        
        # Verify checksum if provided
        if [ -n "$expected_checksum" ]; then
            echo "Verifying checksum for ${model_name}..."
            actual_checksum=$(sha256sum "$local_path" | cut -d' ' -f1)
            
            if [ "$expected_checksum" = "$actual_checksum" ]; then
                echo "✓ Checksum verified for ${model_name}"
            else
                echo "✗ Checksum mismatch for ${model_name}!"
                echo "  Expected: $expected_checksum"
                echo "  Actual: $actual_checksum"
                rm -f "$local_path"
                return 1
            fi
        else
            echo "⚠ No checksum provided for ${model_name}, skipping verification"
        fi
    else
        echo "✗ Failed to download ${model_name}"
        return 1
    fi
}

# Track overall success
all_success=true

# Download each model
for model_name in "${!MODELS[@]}"; do
    s3_path="${MODELS[$model_name]}"
    checksum="${CHECKSUMS[$model_name]}"
    
    if download_and_verify_model "$model_name" "$s3_path" "$checksum"; then
        echo "✅ ${model_name} ready"
    else
        echo "❌ ${model_name} failed"
        all_success=false
        # Continue downloading other models even if one fails
    fi
    echo ""
done

# Create a marker file to indicate models have been downloaded
if $all_success; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${MODEL_DIR}/.models_downloaded"
    echo "✅ All models downloaded successfully"
    exit 0
else
    echo "❌ Some models failed to download"
    # Exit with error if any model failed
    exit 1
fi