# Model Signing and Verification Guide

This document describes the model signing and verification process for the Clarity Loop Backend, ensuring the integrity and authenticity of ML model files.

## Overview

The model signing system uses HMAC-SHA256 to create cryptographic signatures for ML model files. This ensures:

- **Integrity**: Models haven't been tampered with or corrupted
- **Authenticity**: Models come from a trusted source
- **Traceability**: Each signature includes metadata about when it was created

## Quick Start

### Sign a Model

```bash
# Using environment variable (recommended)
export MODEL_SIGNING_KEY="your-secret-key"
python scripts/sign_model.py sign --model models/pat/model.pth

# Sign all models in a directory
python scripts/sign_model.py sign-all --models-dir models/
```

### Verify a Model

```bash
# Verify a single model
python scripts/sign_model.py verify --model models/pat/model.pth

# Verify all models in a directory
python scripts/sign_model.py verify-all --models-dir models/
```

## CI/CD Integration

### GitHub Actions Workflow

The project includes a reusable workflow for automated model signing. To use it in your CI/CD pipeline:

```yaml
name: Build and Sign Models

on:
  push:
    branches: [main]
  pull_request:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      # ... your model building steps ...
      
  sign-models:
    needs: build
    uses: ./.github/workflows/model-signing.yml
    with:
      models_directory: 'models/'
      verify_only: false
    secrets:
      MODEL_SIGNING_KEY: ${{ secrets.MODEL_SIGNING_KEY }}
```

### Setting Up Secrets

1. Go to your GitHub repository settings
2. Navigate to Secrets and variables → Actions
3. Create a new secret named `MODEL_SIGNING_KEY`
4. Use a strong, randomly generated key (recommended: 32+ characters)

### Verification-Only Mode

For pull requests or untrusted environments, use verification-only mode:

```yaml
sign-models:
  uses: ./.github/workflows/model-signing.yml
  with:
    models_directory: 'models/'
    verify_only: true
  secrets:
    MODEL_SIGNING_KEY: ${{ secrets.MODEL_SIGNING_KEY }}
```

## File Structure

When you sign a model, the following files are created:

```
models/
├── pat/
│   ├── model.pth              # Original model file
│   ├── model.pth.sig          # Signature file
│   └── config.json
│       └── config.json.sig    # Each file gets its own signature
└── model_signatures.json      # Manifest of all signatures
```

### Signature File Format

Each `.sig` file contains:

```json
{
  "file": "model.pth",
  "signature": "a3f2b1c4d5e6...",
  "algorithm": "sha256",
  "signed_at": "2025-01-12T10:30:00Z",
  "file_size": 134217728
}
```

### Manifest File Format

The `model_signatures.json` manifest contains:

```json
{
  "created_at": "2025-01-12T10:30:00Z",
  "algorithm": "sha256",
  "total_files": 3,
  "signatures": {
    "pat/model.pth": {
      "file": "model.pth",
      "signature": "a3f2b1c4d5e6...",
      "algorithm": "sha256",
      "signed_at": "2025-01-12T10:30:00Z",
      "file_size": 134217728
    }
  }
}
```

## Security Best Practices

### Key Management

1. **Never commit keys**: Always use environment variables or secrets management
2. **Rotate keys regularly**: Update MODEL_SIGNING_KEY periodically
3. **Use strong keys**: Minimum 32 characters, randomly generated
4. **Limit key access**: Only CI/CD systems should have signing keys

### Signature Storage

1. **Version control signatures**: Commit `.sig` files alongside models
2. **Backup signatures**: Store signature artifacts for audit trails
3. **Verify on deployment**: Always verify signatures before using models

### CI/CD Security

1. **Protected branches**: Only allow signed models on main/production branches
2. **Automated verification**: Verify all models in CI pipeline
3. **Fail on verification errors**: Block deployments if signatures don't match

## Integration with Existing Systems

### With Model Integrity System

The signing system complements the existing checksum-based integrity system:

```python
# Use both systems for defense in depth
from clarity.ml.model_integrity import ModelChecksumManager

# Checksum for integrity
manager = ModelChecksumManager("models/pat")
manager.verify_model_integrity("model_v1")

# HMAC for authenticity
# (handled by sign_model.py in CI/CD)
```

### In Application Startup

Add signature verification to your startup routine:

```python
import subprocess
import sys

def verify_model_signatures():
    """Verify all model signatures on startup."""
    result = subprocess.run(
        ["python", "scripts/sign_model.py", "verify-all", "--models-dir", "models/"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Model signature verification failed!")
        print(result.stderr)
        sys.exit(1)
        
# Call during application initialization
verify_model_signatures()
```

## Troubleshooting

### Common Issues

1. **"Signature file not found"**
   - Ensure models have been signed first
   - Check if `.sig` files exist alongside model files

2. **"Signature mismatch"**
   - Model file has been modified after signing
   - Wrong signing key is being used
   - File corruption during transfer

3. **"MODEL_SIGNING_KEY not set"**
   - Set the environment variable before running
   - In CI/CD, ensure secret is properly configured

### Debugging

Enable verbose mode for detailed output:

```bash
python scripts/sign_model.py verify-all --models-dir models/ --verbose
```

### Recovery

If signatures are lost or corrupted:

1. Restore from backup/artifacts if available
2. Re-sign models with the correct key
3. Update deployment to use new signatures

## Advanced Usage

### Custom File Patterns

By default, the tool signs these file types:
- `*.pth`, `*.pt` (PyTorch)
- `*.onnx` (ONNX)
- `*.pb` (TensorFlow/Protocol Buffers)
- `*.h5` (Keras/HDF5)
- `*.safetensors` (Safetensors format)

To add custom patterns, modify the `model_patterns` list in `sign_model.py`.

### Programmatic Usage

```python
from scripts.sign_model import ModelSigner

# Create signer
signer = ModelSigner(secret_key="your-secret-key")

# Sign a model
metadata = signer.sign_model(Path("models/my_model.pth"))

# Verify a model
is_valid, error = signer.verify_model(Path("models/my_model.pth"))
if not is_valid:
    print(f"Verification failed: {error}")
```

### Batch Operations

For large model repositories:

```bash
# Sign only non-recursive (top-level files only)
python scripts/sign_model.py sign-all --models-dir models/ --no-recursive

# Parallel signing (using GNU parallel)
find models/ -name "*.pth" | parallel -j 4 python scripts/sign_model.py sign --model {}
```

## Monitoring and Auditing

### Signature Age Monitoring

Check when models were last signed:

```bash
# Find signatures older than 30 days
find models/ -name "*.sig" -mtime +30 -exec ls -la {} \;
```

### Audit Log Integration

The signing tool logs all operations. Integrate with your logging system:

```python
import logging

# Configure centralized logging
logging.basicConfig(
    handlers=[
        logging.FileHandler("/var/log/model-signing.log"),
        # Add your log aggregation handler here
    ]
)
```

### Metrics and Alerts

Monitor key metrics:
- Number of models without signatures
- Failed verification attempts
- Signature age distribution
- CI/CD signing success rate

## Migration Guide

If you have existing models without signatures:

1. **Inventory models**: List all model files
   ```bash
   find models/ -type f \( -name "*.pth" -o -name "*.onnx" \) > models_to_sign.txt
   ```

2. **Batch sign**: Sign all existing models
   ```bash
   export MODEL_SIGNING_KEY="your-secret-key"
   python scripts/sign_model.py sign-all --models-dir models/
   ```

3. **Verify signatures**: Ensure all models are properly signed
   ```bash
   python scripts/sign_model.py verify-all --models-dir models/
   ```

4. **Update CI/CD**: Add signing workflow to your pipeline

5. **Deploy verification**: Update production to verify signatures

## Compliance and Standards

This signing system helps meet various compliance requirements:

- **HIPAA**: Ensures integrity of ML models processing health data
- **SOC 2**: Provides audit trail for model changes
- **ISO 27001**: Implements cryptographic controls for data integrity
- **FDA regulations**: Supports validation of AI/ML medical devices

Always consult with your compliance team for specific requirements.