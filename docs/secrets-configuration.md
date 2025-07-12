# Secrets Configuration Guide

This guide explains how to configure secrets for the Clarity Loop Backend in production environments.

## Overview

The Clarity Loop Backend uses a secure parameter store integration for managing sensitive configuration values. The system supports two modes:

1. **AWS Systems Manager (SSM) Parameter Store** - Primary method for production
2. **Environment Variables** - Fallback method and for local development

## Configuration Options

### Environment Variables

The following environment variables control the secrets manager behavior:

- `CLARITY_USE_SSM` - Set to "true" to enable SSM Parameter Store (default: auto-detect)
- `CLARITY_SSM_PREFIX` - SSM parameter prefix (default: "/clarity/production")
- `AWS_DEFAULT_REGION` - AWS region for SSM (default: "us-east-1")
- `CLARITY_SECRETS_CACHE_TTL` - Cache TTL in seconds (default: 300)

### Model Integrity Secrets

For model integrity verification, configure these secrets:

1. **Model Signature Key**
   - SSM Parameter: `{prefix}/model_signature_key`
   - Environment Variable: `MODEL_SIGNATURE_KEY`
   - Purpose: HMAC key for model checksum verification

2. **Expected Model Checksums**
   - SSM Parameter: `{prefix}/expected_model_checksums`
   - Environment Variable: `EXPECTED_MODEL_CHECKSUMS`
   - Format: JSON object mapping model size to SHA-256 checksum

## Production Setup

### 1. AWS SSM Parameter Store Setup

```bash
# Set the model signature key
aws ssm put-parameter \
  --name "/clarity/production/model_signature_key" \
  --value "your-secure-hmac-key-here" \
  --type "SecureString" \
  --description "HMAC key for PAT model integrity verification"

# Set the expected model checksums
aws ssm put-parameter \
  --name "/clarity/production/expected_model_checksums" \
  --value '{
    "small": "4b30d57febbbc8ef221e4b196bf6957e7c7f366f6b836fe800a43f69d24694ad",
    "medium": "6175021ca1a43f3c834bdaa644c45f27817cf985d8ffd186fab9b5de2c4ca661",
    "large": "c93b723f297f0d9d2ad982320b75e9212882c8f38aa40df1b600e9b2b8aa1973"
  }' \
  --type "SecureString" \
  --description "Expected SHA-256 checksums for PAT models"
```

### 2. IAM Permissions

Ensure your application's IAM role has the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ssm:GetParameter",
        "ssm:GetParameters",
        "ssm:DescribeParameters"
      ],
      "Resource": [
        "arn:aws:ssm:*:*:parameter/clarity/production/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt"
      ],
      "Resource": [
        "arn:aws:kms:*:*:key/*"
      ],
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "ssm.*.amazonaws.com"
        }
      }
    }
  ]
}
```

### 3. Environment-Specific Configuration

For different environments, use different SSM prefixes:

```bash
# Development
export CLARITY_SSM_PREFIX="/clarity/development"

# Staging
export CLARITY_SSM_PREFIX="/clarity/staging"

# Production
export CLARITY_SSM_PREFIX="/clarity/production"
```

## Local Development

For local development without AWS access:

```bash
# Disable SSM
export CLARITY_USE_SSM=false

# Set secrets via environment variables
export MODEL_SIGNATURE_KEY="dev-test-key-only"
export EXPECTED_MODEL_CHECKSUMS='{
  "small": "dev-checksum-small",
  "medium": "dev-checksum-medium",
  "large": "dev-checksum-large"
}'
```

## Docker Configuration

When running in Docker:

```dockerfile
# Production Dockerfile
ENV CLARITY_USE_SSM=true
ENV CLARITY_SSM_PREFIX=/clarity/production
ENV AWS_DEFAULT_REGION=us-east-1
```

For local Docker development:

```yaml
# docker-compose.yml
services:
  app:
    environment:
      - CLARITY_USE_SSM=false
      - MODEL_SIGNATURE_KEY=dev-key
      - EXPECTED_MODEL_CHECKSUMS={"small":"dev","medium":"dev","large":"dev"}
```

## Kubernetes Configuration

For Kubernetes deployments:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: clarity-config
data:
  CLARITY_USE_SSM: "true"
  CLARITY_SSM_PREFIX: "/clarity/production"
  AWS_DEFAULT_REGION: "us-east-1"
---
apiVersion: v1
kind: Secret
metadata:
  name: clarity-secrets
type: Opaque
stringData:
  # Fallback values if SSM is unavailable
  MODEL_SIGNATURE_KEY: "fallback-key"
  EXPECTED_MODEL_CHECKSUMS: |
    {
      "small": "fallback-checksum",
      "medium": "fallback-checksum",
      "large": "fallback-checksum"
    }
```

## Testing

To verify secrets configuration:

```python
from clarity.security.secrets_manager import get_secrets_manager

# Check health
manager = get_secrets_manager()
health = manager.health_check()
print(health)

# Verify secrets are accessible
signature_key = manager.get_model_signature_key()
checksums = manager.get_model_checksums()
print(f"Signature key configured: {bool(signature_key)}")
print(f"Checksums configured: {len(checksums)} models")
```

## Security Best Practices

1. **Never commit secrets to version control**
2. **Use SecureString type in SSM for sensitive values**
3. **Rotate signature keys periodically**
4. **Use different keys for different environments**
5. **Monitor parameter access via CloudTrail**
6. **Implement least-privilege IAM policies**
7. **Enable parameter store encryption at rest**

## Troubleshooting

### SSM Connection Issues

If SSM parameters cannot be retrieved:

1. Check IAM permissions
2. Verify AWS credentials are available
3. Check network connectivity to AWS
4. Verify parameter names and prefix
5. Check CloudTrail logs for access attempts

### Fallback Behavior

If SSM is unavailable, the system will:

1. Log a warning
2. Attempt to use environment variables
3. Use hardcoded defaults as last resort
4. Continue operating with reduced security

### Cache Issues

To force refresh of cached secrets:

```python
from clarity.security.secrets_manager import get_secrets_manager

manager = get_secrets_manager()
# Refresh specific key
manager.refresh_cache("model_signature_key")
# Or refresh all
manager.refresh_cache()
```