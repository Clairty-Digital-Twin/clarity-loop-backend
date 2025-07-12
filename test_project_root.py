#!/usr/bin/env python3
"""Test what _PROJECT_ROOT would be in different environments."""

from pathlib import Path
import os

# Simulate the logic from pat_service.py
if os.path.exists("/models/pat"):
    # Docker environment - use mounted volume
    _PROJECT_ROOT = Path("/")
else:
    # Local development - use project root
    _PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # Go up to project root

print(f"Current file: {__file__}")
print(f"_PROJECT_ROOT: {_PROJECT_ROOT}")
print(f"Model path would be: {_PROJECT_ROOT / 'models' / 'pat' / 'PAT-S_29k_weight_transformer.h5'}")

# Test with resolved path
resolved_path = Path(__file__).resolve()
print(f"\nResolved current file: {resolved_path}")
print(f"Parent chain:")
print(f"  1 parent: {resolved_path.parent}")
print(f"  2 parents: {resolved_path.parent.parent}")
print(f"  3 parents: {resolved_path.parent.parent.parent}")
print(f"  4 parents: {resolved_path.parent.parent.parent.parent}")

# Test what happens in production
print("\n--- In production (Docker) ---")
print(f"If /models/pat exists, _PROJECT_ROOT = /")
print(f"Model path would be: /models/pat/PAT-S_29k_weight_transformer.h5")

print("\n--- What might be happening ---")
print("If the file is installed in site-packages, the path calculation changes:")
site_packages_path = Path("/usr/local/lib/python3.11/site-packages/clarity/ml/pat_service.py")
print(f"File at: {site_packages_path}")
print(f"4 parents up: {site_packages_path.parent.parent.parent.parent}")
print(f"Result: {site_packages_path.parent.parent.parent.parent / 'models' / 'pat' / 'PAT-S_29k_weight_transformer.h5'}")