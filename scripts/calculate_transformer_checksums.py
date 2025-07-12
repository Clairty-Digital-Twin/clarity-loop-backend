#!/usr/bin/env python3
"""Calculate checksums for the actual transformer model files from S3."""

import hashlib
import hmac
import os
import boto3
from pathlib import Path
import json
from datetime import datetime

# Model signature key - should match the one in pat_service.py
MODEL_SIGNATURE_KEY = "pat_model_integrity_key_2025"

def calculate_file_checksum(filepath: str) -> tuple[str, str]:
    """Calculate both SHA-256 and HMAC-SHA256 checksums for a file.
    
    Returns:
        Tuple of (sha256_hex, hmac_hex)
    """
    try:
        sha256_hash = hashlib.sha256()
        
        # Read file in chunks to handle large files
        with Path(filepath).open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        
        # Get raw SHA-256
        raw_sha256 = sha256_hash.hexdigest()
        
        # Create HMAC signature (matching pat_service.py logic)
        hmac_signature = hmac.new(
            MODEL_SIGNATURE_KEY.encode("utf-8"),
            raw_sha256.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        
        return raw_sha256, hmac_signature
        
    except Exception as e:
        print(f"Error calculating checksum for {filepath}: {e}")
        return "", ""

def download_and_checksum_models():
    """Download models from S3 and calculate their checksums."""
    s3_bucket = "clarity-ml-models-124355672559"
    models = {
        "small": "pat/PAT-S_29k_weight_transformer.h5",
        "medium": "pat/PAT-M_29k_weight_transformer.h5", 
        "large": "pat/PAT-L_91k_weight_transformer.h5"
    }
    
    # Create temp directory
    temp_dir = Path("/tmp/pat_models_checksum")
    temp_dir.mkdir(exist_ok=True)
    
    # Initialize S3 client
    s3 = boto3.client('s3', region_name='us-east-1')
    
    checksums = {}
    
    for size, s3_key in models.items():
        local_file = temp_dir / Path(s3_key).name
        
        print(f"\n📥 Downloading {size} model from s3://{s3_bucket}/{s3_key}...")
        
        try:
            # Download file
            s3.download_file(s3_bucket, s3_key, str(local_file))
            file_size = local_file.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ Downloaded {local_file.name} ({file_size:.2f} MB)")
            
            # Calculate checksums
            raw_sha256, hmac_sha256 = calculate_file_checksum(str(local_file))
            
            checksums[size] = {
                "filename": Path(s3_key).name,
                "s3_path": f"s3://{s3_bucket}/{s3_key}",
                "file_size_mb": round(file_size, 2),
                "sha256": raw_sha256,
                "hmac_sha256": hmac_sha256
            }
            
            print(f"📊 SHA-256: {raw_sha256}")
            print(f"🔐 HMAC-SHA256: {hmac_sha256}")
            
        except Exception as e:
            print(f"❌ Error processing {size} model: {e}")
            checksums[size] = {
                "error": str(e)
            }
    
    # Save results
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_signature_key": MODEL_SIGNATURE_KEY,
        "checksums": checksums
    }
    
    output_file = Path("pat_transformer_checksums.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    
    # Generate Python code for pat_service.py
    print("\n📝 Update pat_service.py with these checksums:")
    print("```python")
    print("EXPECTED_MODEL_CHECKSUMS = {")
    for size, data in checksums.items():
        if "hmac_sha256" in data:
            print(f'    "{size}": "{data["hmac_sha256"]}",  # {data["filename"]} ({data["file_size_mb"]} MB)')
        else:
            print(f'    "{size}": "",  # Error: {data.get("error", "Unknown")}')
    print("}")
    print("```")
    
    # Cleanup
    print("\n🧹 Cleaning up temporary files...")
    for file in temp_dir.glob("*.h5"):
        file.unlink()
    temp_dir.rmdir()
    
    return checksums

def verify_local_models():
    """Verify checksums of locally downloaded models."""
    model_dirs = [
        Path("/app/models/pat"),
        Path("/models/pat"),
        Path("models/pat")
    ]
    
    print("\n🔍 Checking for locally downloaded models...")
    
    for model_dir in model_dirs:
        if model_dir.exists():
            print(f"\n📁 Found models in {model_dir}:")
            for model_file in model_dir.glob("*.h5"):
                _, hmac_checksum = calculate_file_checksum(str(model_file))
                file_size = model_file.stat().st_size / (1024 * 1024)
                print(f"  - {model_file.name}: {hmac_checksum} ({file_size:.2f} MB)")

if __name__ == "__main__":
    print("🔐 PAT Transformer Model Checksum Calculator")
    print("=" * 50)
    
    # Check if we're in AWS environment
    if os.environ.get("AWS_EXECUTION_ENV") or os.path.exists("/app"):
        print("🏃 Running in AWS/Docker environment")
        verify_local_models()
    else:
        print("💻 Running in local environment")
        
        # Check for AWS credentials
        if os.environ.get("AWS_ACCESS_KEY_ID") or os.path.exists(os.path.expanduser("~/.aws/credentials")):
            print("✅ AWS credentials found")
            download_and_checksum_models()
        else:
            print("❌ AWS credentials not found. Please configure AWS CLI first.")
            print("   Run: aws configure")