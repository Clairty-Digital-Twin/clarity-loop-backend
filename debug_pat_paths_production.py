#!/usr/bin/env python3
"""Debug script to verify PAT model path resolution in production."""

import os
import sys
from pathlib import Path

print("=== PAT Model Path Debug ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Check environment variable
pat_model_base = os.environ.get("PAT_MODEL_BASE_DIR")
print(f"\nPAT_MODEL_BASE_DIR env var: {pat_model_base}")

# Check various possible paths
paths_to_check = [
    "/app/models/pat",
    "/models/pat",
    "/usr/local/lib/python3.11/models/pat",
    "/opt/clarity/models/pat",
    Path.home() / ".clarity" / "models" / "pat"
]

print("\nChecking model directories:")
for path in paths_to_check:
    path = Path(path)
    exists = path.exists()
    print(f"  {path}: {'EXISTS' if exists else 'NOT FOUND'}")
    if exists and path.is_dir():
        try:
            files = list(path.glob("*.h5"))
            if files:
                print(f"    Found {len(files)} .h5 files:")
                for f in files[:5]:  # Show first 5
                    print(f"      - {f.name}")
        except Exception as e:
            print(f"    Error listing files: {e}")

# Test the logic from pat_service.py
print("\n=== Testing PAT Service Logic ===")

# Import the actual module to see where it's installed
try:
    import clarity.ml.pat_service
    pat_service_path = Path(clarity.ml.pat_service.__file__)
    print(f"pat_service.py location: {pat_service_path}")
    print(f"4 parents up: {pat_service_path.parent.parent.parent.parent}")
    
    # Show what _PROJECT_ROOT would be
    if os.environ.get("PAT_MODEL_BASE_DIR"):
        project_root = Path(os.environ["PAT_MODEL_BASE_DIR"]).parent.parent
        print(f"\nUsing PAT_MODEL_BASE_DIR: _PROJECT_ROOT = {project_root}")
    elif os.path.exists("/app/models/pat"):
        project_root = Path("/app")
        print(f"\nUsing Docker production: _PROJECT_ROOT = {project_root}")
    elif os.path.exists("/models/pat"):
        project_root = Path("/")
        print(f"\nUsing alternative Docker: _PROJECT_ROOT = {project_root}")
    else:
        project_root = pat_service_path.parent.parent.parent.parent
        print(f"\nUsing relative path: _PROJECT_ROOT = {project_root}")
    
    model_path = project_root / "models" / "pat" / "PAT-S_29k_weight_transformer.h5"
    print(f"Final model path: {model_path}")
    print(f"Model exists: {model_path.exists()}")
    
except ImportError as e:
    print(f"Could not import clarity.ml.pat_service: {e}")

# Check if models were downloaded
print("\n=== Model Download Status ===")
marker_files = [
    "/app/models/pat/.models_downloaded",
    "/models/pat/.models_downloaded"
]

for marker in marker_files:
    if os.path.exists(marker):
        print(f"✓ Models downloaded marker found: {marker}")
        try:
            with open(marker, 'r') as f:
                print(f"  Downloaded at: {f.read().strip()}")
        except:
            pass
    else:
        print(f"✗ No marker at: {marker}")

print("\n=== Recommendations ===")
if not any(Path(p).exists() for p in ["/app/models/pat", "/models/pat"]):
    print("⚠️ No model directories found!")
    print("1. Ensure download_models.sh ran successfully")
    print("2. Check ECS task logs for download errors")
    print("3. Verify S3 permissions and bucket access")
else:
    print("✓ Model directory exists")
    if pat_model_base:
        print(f"✓ PAT_MODEL_BASE_DIR is set to: {pat_model_base}")
    else:
        print("💡 Consider setting PAT_MODEL_BASE_DIR=/app/models/pat in ECS task definition")