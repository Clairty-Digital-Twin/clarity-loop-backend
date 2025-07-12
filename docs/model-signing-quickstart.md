# Model Signing Quick Start Guide

## 🚀 Quick Setup for CI/CD

### 1. Set up GitHub Secret

```bash
# Generate a strong secret key
openssl rand -base64 32

# Add to GitHub Secrets:
# Settings → Secrets → Actions → New repository secret
# Name: MODEL_SIGNING_KEY
# Value: [your generated key]
```

### 2. Add to Your CI/CD Pipeline

```yaml
# .github/workflows/your-pipeline.yml
jobs:
  sign-models:
    uses: ./.github/workflows/model-signing.yml
    with:
      models_directory: 'models/'
    secrets:
      MODEL_SIGNING_KEY: ${{ secrets.MODEL_SIGNING_KEY }}
```

### 3. Verify on Startup

```python
# In your main application file
from scripts.verify_models_startup import startup_verification

if not startup_verification():
    print("Model verification failed!")
    sys.exit(1)
```

## 🔧 Common Commands

```bash
# Sign a single model
export MODEL_SIGNING_KEY="your-secret-key"
python scripts/sign_model.py sign --model models/my_model.pth

# Sign all models
python scripts/sign_model.py sign-all --models-dir models/

# Verify signatures
python scripts/sign_model.py verify-all --models-dir models/
```

## 📁 File Structure

```
models/
├── model.pth          # Original model
├── model.pth.sig      # Signature file
└── model_signatures.json  # Manifest
```

## 🔒 Security Checklist

- [ ] Secret key stored in GitHub Secrets
- [ ] Never commit secret keys
- [ ] Verify signatures before deployment
- [ ] Sign models only in trusted environments
- [ ] Monitor signature age (rotate if > 90 days)

## 🚨 Troubleshooting

| Error | Solution |
|-------|----------|
| "Signature file not found" | Run sign command first |
| "Signature mismatch" | Model was modified or wrong key |
| "MODEL_SIGNING_KEY not set" | Export the environment variable |

## 📊 CI/CD Status Checks

```yaml
# Ensure models are signed before merge
- name: Check Model Signatures
  run: |
    python scripts/sign_model.py verify-all --models-dir models/
```

## 🔄 Integration Examples

### Docker
```dockerfile
# Verify during build
RUN MODEL_SIGNING_KEY=${MODEL_SIGNING_KEY} \
    python scripts/sign_model.py verify-all --models-dir models/
```

### Kubernetes
```yaml
# Init container for verification
initContainers:
  - name: verify-models
    env:
      - name: MODEL_SIGNING_KEY
        valueFrom:
          secretKeyRef:
            name: model-signing
            key: signing-key
    command: ["python", "scripts/sign_model.py", "verify-all"]
```

## 📝 Best Practices

1. **Development**: Use `--development` flag to skip verification
2. **Staging**: Verify signatures but allow warnings
3. **Production**: Strict verification, fail on any error

## 🔗 Related Documentation

- [Full Model Signing Guide](model-signing.md)
- [Security Best Practices](../security/README.md)
- [CI/CD Pipeline Guide](../ci-cd/README.md)