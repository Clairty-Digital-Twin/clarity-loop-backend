# Clarity Loop Backend Scripts

This directory contains utility scripts for development, deployment, and maintenance of the Clarity Loop Backend.

## Model Security Scripts

### 🔐 sign_model.py
**Purpose**: Sign and verify ML model files using HMAC-SHA256 for integrity and authenticity.

```bash
# Sign a model
python sign_model.py sign --model models/pat/model.pth

# Verify a model
python sign_model.py verify --model models/pat/model.pth

# Sign all models in a directory
python sign_model.py sign-all --models-dir models/

# Verify all models
python sign_model.py verify-all --models-dir models/
```

**CI/CD Integration**: Use the `MODEL_SIGNING_KEY` environment variable for automated signing.

### ✅ verify_models_startup.py
**Purpose**: Comprehensive model verification for application startup, combining signature and checksum verification.

```bash
# Full verification (production)
python verify_models_startup.py

# Development mode (warnings only)
python verify_models_startup.py --development

# Skip signature verification
python verify_models_startup.py --no-signatures
```

### 🔍 model_integrity_cli.py
**Purpose**: Manage model checksums for integrity verification.

```bash
# Register a model
python model_integrity_cli.py register model_name --models-dir models/

# Verify model integrity
python model_integrity_cli.py verify --model model_name

# List registered models
python model_integrity_cli.py list
```

## Model Management Scripts

### 📦 create_placeholder_models.py
**Purpose**: Create placeholder model files for testing and development.

```bash
python create_placeholder_models.py
```

### 🔢 calculate_model_checksums.py
**Purpose**: Calculate and display checksums for model files.

```bash
python calculate_model_checksums.py
```

### ⬇️ download_models.sh
**Purpose**: Download ML models from secure storage.

```bash
./download_models.sh
```

## Deployment Scripts

### 🚀 entrypoint.sh
**Purpose**: Production Docker entrypoint script with health checks and initialization.

### 🐛 entrypoint-debug.sh
**Purpose**: Debug version of entrypoint with verbose logging and development features.

### 🔧 startup_validator.py
**Purpose**: Validate application startup configuration and dependencies.

```bash
python startup_validator.py
```

## Security Testing Scripts

### 🛡️ test_security_headers.py
**Purpose**: Test security headers in API responses.

```bash
python test_security_headers.py
```

### 🔒 test_lockout.py
**Purpose**: Test account lockout mechanisms.

```bash
python test_lockout.py
```

### ⏱️ test_rate_limiting.py
**Purpose**: Test API rate limiting functionality.

```bash
python test_rate_limiting.py
```

### 🧪 smoke-test-auth-suite.sh
**Purpose**: Comprehensive authentication system smoke tests.

```bash
./smoke-test-auth-suite.sh
```

### 🔍 smoke-test-main.sh
**Purpose**: Main application smoke tests.

```bash
./smoke-test-main.sh
```

## API Documentation Scripts

### 📄 generate_openapi.py
**Purpose**: Generate OpenAPI specification from FastAPI application.

```bash
python generate_openapi.py
```

### 🧹 clean_openapi.py
**Purpose**: Clean and format OpenAPI specification.

```bash
python clean_openapi.py
```

### ✅ validate_openapi.sh
**Purpose**: Validate OpenAPI specification against standards.

```bash
./validate_openapi.sh
```

## Utility Scripts

### 🔑 store-gcp-credentials.sh
**Purpose**: Store GCP credentials securely.

```bash
./store-gcp-credentials.sh
```

### 🔐 test-secrets.py
**Purpose**: Test secret management functionality.

```bash
python test-secrets.py
```

### 🤖 clarity-models
**Purpose**: Model management CLI tool.

```bash
./clarity-models --help
```

## Best Practices

1. **Security**: Always use environment variables for sensitive data
2. **Permissions**: Ensure scripts have appropriate execute permissions
3. **Documentation**: Update this README when adding new scripts
4. **Testing**: Test scripts in development before production use
5. **Logging**: Use appropriate logging levels for different environments

## Environment Variables

Common environment variables used by scripts:

- `MODEL_SIGNING_KEY`: Secret key for model signing
- `ENVIRONMENT`: Current environment (development/staging/production)
- `LOG_LEVEL`: Logging verbosity
- `AWS_REGION`: AWS region for deployments
- `GCP_PROJECT`: GCP project ID

## Contributing

When adding new scripts:

1. Include comprehensive `--help` documentation
2. Add error handling and logging
3. Follow Python/Bash best practices
4. Update this README
5. Add tests if applicable