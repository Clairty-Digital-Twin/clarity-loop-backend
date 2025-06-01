# Build Process Checklist

**Canonical step-by-step build verification for healthcare AI systems**

## Core Philosophy

**Systematic Verification**: Every build step must be verifiable, traceable, and compliant with healthcare regulations. No assumptions - every component is validated before proceeding to the next phase.

**Risk-Stratified Deployment**: Critical healthcare components (authentication, data encryption, ML models) require additional verification steps compared to utility functions.

**Zero-Downtime Philosophy**: Build processes must support continuous deployment without affecting patient data access or clinical workflows.

## Build Process Overview

### Phase Sequence
1. **Environment Setup & Validation** (0-10 minutes)
2. **Dependency Installation & Security Audit** (5-15 minutes)
3. **Code Quality & Security Verification** (2-5 minutes)
4. **Test Suite Execution** (10-30 minutes)
5. **ML Model Validation** (5-15 minutes)
6. **HIPAA Compliance Verification** (3-10 minutes)
7. **Build & Package Creation** (5-10 minutes)
8. **Deployment Readiness Check** (2-5 minutes)
9. **Production Deployment** (10-20 minutes)
10. **Post-Deployment Verification** (5-15 minutes)

**Total Build Time**: 45-135 minutes (varies by complexity and test coverage)

## Phase 1: Environment Setup & Validation

### 1.1 Python Environment Verification

**Checklist**:
- [ ] Python 3.11+ installed and accessible
- [ ] UV package manager installed and updated
- [ ] Virtual environment created and activated
- [ ] Environment variables properly configured
- [ ] Google Cloud SDK installed and authenticated

**Commands**:
```bash
# Verify Python version
python --version  # Must be 3.11+

# Verify UV installation
uv --version

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Verify UV environment
uv python list

# Test Google Cloud authentication
gcloud auth list
gcloud config get-value project
```

**Validation Script**:
```bash
#!/bin/bash
# scripts/verify-environment.sh

echo "🔍 Environment Setup Verification"
echo "================================="

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
REQUIRED_VERSION="3.11.0"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
    echo "✅ Python $PYTHON_VERSION (>= $REQUIRED_VERSION)"
else
    echo "❌ Python $PYTHON_VERSION is below required $REQUIRED_VERSION"
    exit 1
fi

# Check UV
if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version)
    echo "✅ UV: $UV_VERSION"
else
    echo "❌ UV package manager not found"
    exit 1
fi

# Check virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment: $VIRTUAL_ENV"
else
    echo "❌ Virtual environment not activated"
    exit 1
fi

# Check Google Cloud CLI
if command -v gcloud &> /dev/null; then
    GCLOUD_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null)
    if [[ -n "$GCLOUD_ACCOUNT" ]]; then
        echo "✅ Google Cloud authenticated: $GCLOUD_ACCOUNT"
    else
        echo "❌ Google Cloud not authenticated"
        exit 1
    fi
else
    echo "❌ Google Cloud SDK not found"
    exit 1
fi

echo ""
echo "🎯 Environment setup verified successfully!"
```

### 1.2 Configuration Validation

**Checklist**:
- [ ] `.env` file exists with required variables
- [ ] Firebase configuration valid
- [ ] Google Cloud project accessible
- [ ] PyTorch and CUDA configuration (if using GPU)
- [ ] Redis connection testable

**Environment Variables Required**:
```bash
# .env.example - Required environment variables

# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
GCP_PROJECT_ID="your-project-id"

# Firebase
FIREBASE_PROJECT_ID="your-firebase-project"
FIREBASE_WEB_API_KEY="your-web-api-key"

# Vertex AI
VERTEX_AI_LOCATION="us-central1"
VERTEX_AI_ENDPOINT="your-vertex-endpoint"

# Redis (optional)
REDIS_URL="redis://localhost:6379"

# Development
DEBUG="true"
LOG_LEVEL="INFO"

# HIPAA Compliance
ENCRYPTION_KEY="your-encryption-key-base64"
AUDIT_LOG_LEVEL="ALL"
```

**Configuration Test Script**:
```python
# scripts/test_config.py
import os
import sys
from pathlib import Path
from google.cloud import firestore
from google.cloud import storage
import redis
import logging

def test_environment_variables():
    """Test all required environment variables are present"""
    required_vars = [
        'GOOGLE_APPLICATION_CREDENTIALS',
        'GCP_PROJECT_ID',
        'FIREBASE_PROJECT_ID',
        'FIREBASE_WEB_API_KEY',
        'VERTEX_AI_LOCATION',
        'ENCRYPTION_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    print("✅ All required environment variables present")
    return True

def test_google_cloud_connectivity():
    """Test Google Cloud services connectivity"""
    try:
        # Test Firestore
        db = firestore.Client()
        collections = list(db.collections())
        print("✅ Firestore connection successful")
        
        # Test Cloud Storage
        storage_client = storage.Client()
        buckets = list(storage_client.list_buckets())
        print("✅ Cloud Storage connection successful")
        
        return True
    except Exception as e:
        print(f"❌ Google Cloud connection failed: {e}")
        return False

def test_redis_connectivity():
    """Test Redis connectivity (optional)"""
    try:
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        r = redis.from_url(redis_url)
        r.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"⚠️ Redis connection failed: {e} (optional)")
        return True  # Redis is optional

def main():
    print("🔍 Configuration Validation")
    print("==========================")
    
    success = True
    success &= test_environment_variables()
    success &= test_google_cloud_connectivity()
    success &= test_redis_connectivity()
    
    if success:
        print("\n🎯 Configuration validation successful!")
        sys.exit(0)
    else:
        print("\n💥 Configuration validation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Phase 2: Dependency Installation & Security Audit

### 2.1 Dependency Installation

**Checklist**:
- [ ] Core dependencies installed via UV
- [ ] Development dependencies installed
- [ ] Pre-commit hooks configured
- [ ] No dependency conflicts
- [ ] Security vulnerabilities scanned

**Installation Commands**:
```bash
# Install production dependencies
uv pip install -r requirements.txt

# Install development dependencies
uv pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Verify installation
uv pip list

# Check for dependency conflicts
uv pip check
```

**Dependency Security Audit**:
```bash
#!/bin/bash
# scripts/security-audit.sh

echo "🔒 Dependency Security Audit"
echo "============================"

# Check for known vulnerabilities
echo "Scanning for known vulnerabilities..."
pip-audit

# Check for outdated packages with security issues
echo ""
echo "Checking for outdated packages..."
uv pip list --outdated

# Generate dependency tree
echo ""
echo "Dependency tree analysis..."
pipdeptree

# Check for GPL or other problematic licenses
echo ""
echo "License compliance check..."
pip-licenses --summary

echo ""
echo "🎯 Security audit complete!"
```

### 2.2 ML Model Dependencies

**Checklist**:
- [ ] PyTorch 2.0+ installed with correct CUDA support
- [ ] ONNX Runtime installed and tested
- [ ] Transformers library compatible versions
- [ ] Model files downloadable and accessible
- [ ] GPU memory requirements verified

**ML Dependencies Test**:
```python
# scripts/test_ml_dependencies.py
import torch
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import logging

def test_pytorch_installation():
    """Test PyTorch installation and CUDA availability"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test basic tensor operations
    x = torch.randn(5, 3)
    y = torch.randn(3, 4)
    z = torch.mm(x, y)
    assert z.shape == (5, 4)
    print("✅ PyTorch tensor operations working")
    
    if torch.cuda.is_available():
        x_gpu = x.cuda()
        y_gpu = y.cuda()
        z_gpu = torch.mm(x_gpu, y_gpu)
        assert z_gpu.device.type == 'cuda'
        print("✅ PyTorch CUDA operations working")
    
    return True

def test_onnx_runtime():
    """Test ONNX Runtime installation"""
    print(f"ONNX Runtime version: {ort.__version__}")
    
    # Test basic ONNX session creation
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    
    # Test with a dummy model (this would normally be your actual model)
    # For now, just verify the providers are available
    if 'CUDAExecutionProvider' in providers:
        print("✅ CUDA provider available for ONNX")
    else:
        print("⚠️ CUDA provider not available for ONNX (CPU only)")
    
    print("✅ ONNX Runtime installation verified")
    return True

def test_transformers_library():
    """Test Transformers library functionality"""
    try:
        # Test tokenizer loading (lightweight test)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Test basic tokenization
        text = "This is a test sentence for healthcare AI processing."
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        assert len(tokens) > 0
        assert isinstance(decoded, str)
        
        print("✅ Transformers library working correctly")
        return True
    except Exception as e:
        print(f"❌ Transformers library test failed: {e}")
        return False

def main():
    print("🧠 ML Dependencies Verification")
    print("==============================")
    
    success = True
    try:
        success &= test_pytorch_installation()
        success &= test_onnx_runtime()
        success &= test_transformers_library()
    except Exception as e:
        print(f"❌ ML dependencies test failed: {e}")
        success = False
    
    if success:
        print("\n🎯 ML dependencies verified successfully!")
        return 0
    else:
        print("\n💥 ML dependencies verification failed!")
        return 1

if __name__ == "__main__":
    exit(main())
```

## Phase 3: Code Quality & Security Verification

### 3.1 Code Quality Checks

**Checklist**:
- [ ] MyPy type checking passes (no errors)
- [ ] Ruff linting passes (no violations)
- [ ] Black formatting applied
- [ ] Import sorting correct (isort)
- [ ] Docstring coverage adequate
- [ ] Cyclomatic complexity acceptable

**Quality Check Script**:
```bash
#!/bin/bash
# scripts/quality-check.sh

echo "📊 Code Quality Verification"
echo "==========================="

# Type checking with MyPy
echo "Running MyPy type checking..."
if mypy clarity/; then
    echo "✅ MyPy type checking passed"
else
    echo "❌ MyPy type checking failed"
    exit 1
fi

# Linting with Ruff
echo ""
echo "Running Ruff linting..."
if ruff check clarity/; then
    echo "✅ Ruff linting passed"
else
    echo "❌ Ruff linting failed"
    exit 1
fi

# Code formatting check
echo ""
echo "Checking code formatting..."
if black --check clarity/; then
    echo "✅ Black formatting check passed"
else
    echo "❌ Code formatting issues found"
    echo "Run 'black clarity/' to fix formatting"
    exit 1
fi

# Import sorting check
echo ""
echo "Checking import sorting..."
if isort --check-only clarity/; then
    echo "✅ Import sorting check passed"
else
    echo "❌ Import sorting issues found"
    echo "Run 'isort clarity/' to fix import sorting"
    exit 1
fi

# Docstring coverage
echo ""
echo "Checking docstring coverage..."
if interrogate -v clarity/ --fail-under=80; then
    echo "✅ Docstring coverage adequate (≥80%)"
else
    echo "❌ Docstring coverage below 80%"
    exit 1
fi

echo ""
echo "🎯 Code quality verification complete!"
```

### 3.2 Security Scanning

**Checklist**:
- [ ] Bandit security scanning (no high/medium issues)
- [ ] Secret scanning (no exposed credentials)
- [ ] Dependency vulnerability scanning
- [ ] SAST (Static Application Security Testing)
- [ ] Hardcoded password detection

**Security Scan Script**:
```bash
#!/bin/bash
# scripts/security-scan.sh

echo "🔒 Security Scanning"
echo "==================="

# Bandit security scanning
echo "Running Bandit security scan..."
if bandit -r clarity/ -f json -o bandit-report.json; then
    echo "✅ Bandit security scan passed"
else
    echo "❌ Bandit security issues found"
    echo "Check bandit-report.json for details"
    exit 1
fi

# Secret scanning with detect-secrets
echo ""
echo "Scanning for exposed secrets..."
if detect-secrets scan --all-files --force-use-all-plugins; then
    echo "✅ No secrets detected"
else
    echo "❌ Potential secrets found"
    echo "Review and update .secrets.baseline if needed"
    exit 1
fi

# Check for hardcoded passwords/tokens
echo ""
echo "Scanning for hardcoded credentials..."
if grep -r -n -i "password\|token\|secret\|key" clarity/ --include="*.py" | grep -v "# nosec" | grep -E "(=|:)" | head -10; then
    echo "⚠️ Potential hardcoded credentials found (review required)"
else
    echo "✅ No obvious hardcoded credentials found"
fi

# Additional security checks
echo ""
echo "Additional security validations..."

# Check for debug mode in production code
if grep -r "DEBUG.*=.*True" clarity/ --include="*.py"; then
    echo "⚠️ Debug mode found in code (ensure it's not enabled in production)"
else
    echo "✅ No debug mode found in production code"
fi

# Check for commented-out sensitive code
if grep -r "#.*password\|#.*secret\|#.*key" clarity/ --include="*.py" | head -5; then
    echo "⚠️ Commented-out sensitive code found (review required)"
else
    echo "✅ No commented-out sensitive code found"
fi

echo ""
echo "🎯 Security scanning complete!"
```

## Phase 4: Test Suite Execution

### 4.1 Unit Tests

**Checklist**:
- [ ] All unit tests pass
- [ ] Test coverage ≥90% for core modules
- [ ] No skipped tests in critical paths
- [ ] Test execution time reasonable (<10 minutes)
- [ ] Memory usage acceptable during tests

**Unit Test Execution**:
```bash
#!/bin/bash
# scripts/run-unit-tests.sh

echo "🧪 Unit Test Execution"
echo "====================="

# Run unit tests with coverage
echo "Running unit tests with coverage..."
pytest tests/unit/ \
    --cov=clarity \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    --cov-fail-under=90 \
    --tb=short \
    --durations=10 \
    -v

if [ $? -eq 0 ]; then
    echo "✅ Unit tests passed with coverage ≥90%"
else
    echo "❌ Unit tests failed or coverage below 90%"
    exit 1
fi

# Check for skipped tests
SKIPPED_COUNT=$(pytest tests/unit/ --collect-only -q | grep "skipped" | wc -l)
if [ $SKIPPED_COUNT -gt 0 ]; then
    echo "⚠️ $SKIPPED_COUNT tests are skipped"
    pytest tests/unit/ -v -k "skip" --collect-only
else
    echo "✅ No skipped tests"
fi

echo ""
echo "🎯 Unit tests completed successfully!"
```

### 4.2 Integration Tests

**Checklist**:
- [ ] Database integration tests pass
- [ ] External API integration tests pass
- [ ] Firebase integration tests pass
- [ ] ML model integration tests pass
- [ ] End-to-end workflow tests pass

**Integration Test Script**:
```bash
#!/bin/bash
# scripts/run-integration-tests.sh

echo "🔗 Integration Test Execution"
echo "============================"

# Ensure test environment is ready
echo "Setting up test environment..."
export TEST_ENV=true
export FIRESTORE_EMULATOR_HOST=localhost:8080

# Start Firebase emulator if not running
if ! nc -z localhost 8080; then
    echo "Starting Firebase emulator..."
    firebase emulators:start --only firestore --project demo-test &
    EMULATOR_PID=$!
    sleep 5
fi

# Run integration tests
echo "Running integration tests..."
pytest tests/integration/ \
    --tb=short \
    --durations=10 \
    -v \
    --maxfail=5

INTEGRATION_RESULT=$?

# Clean up
if [ ! -z "$EMULATOR_PID" ]; then
    echo "Stopping Firebase emulator..."
    kill $EMULATOR_PID
fi

if [ $INTEGRATION_RESULT -eq 0 ]; then
    echo "✅ Integration tests passed"
else
    echo "❌ Integration tests failed"
    exit 1
fi

echo ""
echo "🎯 Integration tests completed successfully!"
```

### 4.3 HIPAA Compliance Tests

**Checklist**:
- [ ] Data encryption tests pass
- [ ] Access control tests pass
- [ ] Audit logging tests pass
- [ ] Data retention tests pass
- [ ] Privacy protection tests pass

**HIPAA Compliance Test Script**:
```python
# tests/compliance/test_hipaa_compliance.py
import pytest
import asyncio
from clarity.models import PatientHealthData
from clarity.services import HealthDataService, AuditService
from clarity.security import EncryptionService

class TestHIPAACompliance:
    
    @pytest.mark.asyncio
    async def test_data_encryption_at_rest(self):
        """Verify patient data is encrypted when stored"""
        service = HealthDataService()
        
        patient_data = PatientHealthData(
            patient_id="test_patient_123",
            heart_rate=72,
            timestamp="2024-01-01T12:00:00Z"
        )
        
        # Store data
        doc_id = await service.store_patient_data(patient_data)
        
        # Verify raw storage is encrypted
        raw_storage = await service.get_raw_storage(doc_id)
        raw_content = str(raw_storage)
        
        # Patient ID should not appear in plaintext
        assert "test_patient_123" not in raw_content
        assert "heart_rate" not in raw_content
        
        # But encrypted data should be retrievable
        retrieved_data = await service.retrieve_patient_data(doc_id)
        assert retrieved_data.patient_id == "test_patient_123"
        assert retrieved_data.heart_rate == 72
    
    @pytest.mark.asyncio
    async def test_access_logging(self):
        """Verify all data access is logged for audit"""
        audit_service = AuditService()
        health_service = HealthDataService()
        
        patient_id = "test_patient_audit"
        user_id = "test_doctor_123"
        
        # Access patient data with user context
        with health_service.audit_context(user_id=user_id):
            await health_service.get_patient_data(patient_id)
        
        # Verify audit log entry
        audit_logs = await audit_service.get_logs(patient_id)
        
        assert len(audit_logs) > 0
        latest_log = audit_logs[-1]
        assert latest_log["user_id"] == user_id
        assert latest_log["patient_id"] == patient_id
        assert latest_log["action"] == "data_access"
        assert latest_log["timestamp"] is not None
        assert latest_log["ip_address"] is not None
    
    @pytest.mark.asyncio
    async def test_access_control(self):
        """Verify users can only access authorized data"""
        health_service = HealthDataService()
        
        # User should access their own data
        with health_service.auth_context(user_id="patient_123"):
            data = await health_service.get_patient_data("patient_123")
            assert data is not None
        
        # User should NOT access other patient data
        with pytest.raises(PermissionError):
            with health_service.auth_context(user_id="patient_123"):
                await health_service.get_patient_data("patient_456")
    
    def test_encryption_strength(self):
        """Verify encryption meets HIPAA requirements"""
        encryption_service = EncryptionService()
        
        test_data = "sensitive patient information"
        encrypted = encryption_service.encrypt(test_data)
        
        # Encrypted data should be different from original
        assert encrypted != test_data
        
        # Should be able to decrypt
        decrypted = encryption_service.decrypt(encrypted)
        assert decrypted == test_data
        
        # Verify encryption algorithm strength
        assert encryption_service.algorithm == "AES-256-GCM"
        assert len(encryption_service.key) >= 256 // 8  # 256-bit key
```

## Phase 5: ML Model Validation

### 5.1 PAT Model Validation

**Checklist**:
- [ ] Model loads successfully
- [ ] Model inference works with test data
- [ ] Model outputs are within expected ranges
- [ ] Model performance meets benchmarks
- [ ] Model resource usage acceptable

**ML Model Test Script**:
```python
# scripts/test_ml_models.py
import numpy as np
import torch
import asyncio
from pathlib import Path
from clarity.ml import PATModel, ActigraphyPreprocessor
from clarity.models import ActigraphyData

async def test_pat_model_loading():
    """Test PAT model loading and basic functionality"""
    print("Testing PAT model loading...")
    
    model = PATModel()
    await model.load_model()
    
    assert model.model is not None
    assert model.is_loaded
    print("✅ PAT model loaded successfully")

async def test_pat_model_inference():
    """Test PAT model inference with sample data"""
    print("Testing PAT model inference...")
    
    model = PATModel()
    await model.load_model()
    
    # Generate sample actigraphy data (24 hours, 1-minute intervals)
    sample_data = np.random.randint(0, 2, size=1440)  # Binary activity data
    
    # Test preprocessing
    preprocessor = ActigraphyPreprocessor()
    processed_data = preprocessor.preprocess(sample_data)
    
    assert processed_data.shape == (1440,)
    assert abs(processed_data.mean()) < 0.1  # Should be normalized
    print("✅ Actigraphy preprocessing working")
    
    # Test model inference
    prediction = await model.predict_depression_risk(sample_data)
    
    assert isinstance(prediction, float)
    assert 0.0 <= prediction <= 1.0
    print(f"✅ PAT model inference working (prediction: {prediction:.3f})")

async def test_model_performance():
    """Test model performance with benchmark data"""
    print("Testing model performance...")
    
    model = PATModel()
    await model.load_model()
    
    # Test inference speed
    import time
    sample_data = np.random.randint(0, 2, size=1440)
    
    start_time = time.time()
    for _ in range(10):
        prediction = await model.predict_depression_risk(sample_data)
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / 10
    assert avg_inference_time < 1.0  # Should be under 1 second
    print(f"✅ Model inference speed acceptable ({avg_inference_time:.3f}s avg)")

async def test_model_memory_usage():
    """Test model memory usage"""
    print("Testing model memory usage...")
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    model = PATModel()
    await model.load_model()
    
    loaded_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = loaded_memory - initial_memory
    
    assert memory_increase < 500  # Should use less than 500MB
    print(f"✅ Model memory usage acceptable ({memory_increase:.1f}MB)")

async def main():
    print("🧠 ML Model Validation")
    print("=====================")
    
    try:
        await test_pat_model_loading()
        await test_pat_model_inference()
        await test_model_performance()
        await test_model_memory_usage()
        
        print("\n🎯 ML model validation completed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ ML model validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
```

### 5.2 Model Versioning and Reproducibility

**Checklist**:
- [ ] Model version tracked in metadata
- [ ] Model checksums verified
- [ ] Reproducible inference results
- [ ] Model artifacts properly stored
- [ ] Rollback capability tested

## Phase 6: HIPAA Compliance Verification

### 6.1 Compliance Automation

**Checklist**:
- [ ] Encryption verification automated
- [ ] Access control verification automated
- [ ] Audit logging verification automated
- [ ] Data retention policy enforced
- [ ] Compliance report generated

**HIPAA Compliance Automation**:
```bash
#!/bin/bash
# scripts/hipaa-compliance-check.sh

echo "🏥 HIPAA Compliance Verification"
echo "==============================="

# Run HIPAA compliance tests
echo "Running HIPAA compliance test suite..."
pytest tests/compliance/ -v --tb=short

if [ $? -ne 0 ]; then
    echo "❌ HIPAA compliance tests failed"
    exit 1
fi

# Generate compliance report
echo ""
echo "Generating compliance report..."
python scripts/generate_compliance_report.py

# Verify audit log functionality
echo ""
echo "Testing audit log functionality..."
python scripts/test_audit_logs.py

# Check encryption configuration
echo ""
echo "Verifying encryption configuration..."
python scripts/verify_encryption.py

echo ""
echo "🎯 HIPAA compliance verification complete!"
```

### 6.2 Data Privacy Verification

**Checklist**:
- [ ] PII detection and masking working
- [ ] Data anonymization functional
- [ ] Cross-patient data isolation verified
- [ ] Data export controls working
- [ ] Right to deletion functional

## Phase 7: Build & Package Creation

### 7.1 Application Packaging

**Checklist**:
- [ ] Docker image builds successfully
- [ ] Image size optimized
- [ ] Security vulnerabilities scanned
- [ ] Multi-architecture support
- [ ] Image tagged appropriately

**Docker Build Script**:
```bash
#!/bin/bash
# scripts/build-docker.sh

echo "🐳 Docker Build Process"
echo "======================"

# Build base image
echo "Building Docker image..."
docker build -t clarity-backend:latest .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

# Check image size
IMAGE_SIZE=$(docker images clarity-backend:latest --format "table {{.Size}}" | tail -n 1)
echo "✅ Docker image built successfully (Size: $IMAGE_SIZE)"

# Security scan
echo ""
echo "Scanning Docker image for vulnerabilities..."
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy image clarity-backend:latest

# Test image functionality
echo ""
echo "Testing Docker image..."
docker run --rm clarity-backend:latest python -c "import clarity; print('✅ Package import successful')"

# Tag for deployment
if [ "$CI" = "true" ]; then
    BUILD_ID=${GITHUB_SHA:-$(git rev-parse HEAD)}
    docker tag clarity-backend:latest clarity-backend:$BUILD_ID
    echo "✅ Image tagged with build ID: $BUILD_ID"
fi

echo ""
echo "🎯 Docker build completed successfully!"
```

### 7.2 Artifact Management

**Checklist**:
- [ ] Build artifacts stored securely
- [ ] Artifact integrity verified
- [ ] Deployment manifests generated
- [ ] Configuration templates created
- [ ] Rollback artifacts maintained

## Phase 8: Deployment Readiness Check

### 8.1 Pre-Deployment Validation

**Checklist**:
- [ ] Target environment accessible
- [ ] Database migrations ready
- [ ] Configuration secrets available
- [ ] Load balancer configuration ready
- [ ] Monitoring and alerting configured

**Deployment Readiness Script**:
```bash
#!/bin/bash
# scripts/deployment-readiness.sh

echo "🚀 Deployment Readiness Check"
echo "=============================="

# Check Google Cloud project access
echo "Verifying Google Cloud access..."
gcloud auth list --filter=status:ACTIVE --format="value(account)"
if [ $? -eq 0 ]; then
    echo "✅ Google Cloud authentication verified"
else
    echo "❌ Google Cloud authentication failed"
    exit 1
fi

# Check Cloud Run service exists
SERVICE_NAME="clarity-backend"
REGION="us-central1"

echo ""
echo "Checking Cloud Run service..."
gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(metadata.name)" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Cloud Run service exists"
else
    echo "⚠️ Cloud Run service will be created"
fi

# Check Firestore database
echo ""
echo "Verifying Firestore database..."
python -c "
from google.cloud import firestore
try:
    db = firestore.Client()
    collections = list(db.collections())
    print('✅ Firestore database accessible')
except Exception as e:
    print(f'❌ Firestore access failed: {e}')
    exit(1)
"

# Check required secrets
echo ""
echo "Verifying secrets..."
REQUIRED_SECRETS=("encryption-key" "firebase-config" "jwt-secret")
for secret in "${REQUIRED_SECRETS[@]}"; do
    gcloud secrets describe $secret --format="value(name)" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ Secret '$secret' exists"
    else
        echo "❌ Secret '$secret' missing"
        exit 1
    fi
done

# Test database connectivity
echo ""
echo "Testing database connectivity..."
python scripts/test_db_connectivity.py

echo ""
echo "🎯 Deployment readiness verified!"
```

### 8.2 Configuration Validation

**Checklist**:
- [ ] Environment-specific configs validated
- [ ] Resource limits appropriate
- [ ] Scaling parameters set
- [ ] Health check endpoints configured
- [ ] Logging configuration verified

## Phase 9: Production Deployment

### 9.1 Deployment Execution

**Checklist**:
- [ ] Blue-green deployment strategy
- [ ] Database migrations executed
- [ ] Health checks passing
- [ ] Traffic routing verified
- [ ] Rollback plan ready

**Deployment Script**:
```bash
#!/bin/bash
# scripts/deploy-production.sh

echo "🌐 Production Deployment"
echo "======================="

set -e  # Exit on any error

SERVICE_NAME="clarity-backend"
REGION="us-central1"
PROJECT_ID=$(gcloud config get-value project)
IMAGE_TAG=${1:-latest}

echo "Deploying to project: $PROJECT_ID"
echo "Service: $SERVICE_NAME"
echo "Region: $REGION"
echo "Image tag: $IMAGE_TAG"

# Deploy to Cloud Run
echo ""
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/clarity-backend:$IMAGE_TAG \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-env-vars "ENV=production" \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 100 \
    --timeout 300 \
    --concurrency 1000

if [ $? -eq 0 ]; then
    echo "✅ Cloud Run deployment successful"
else
    echo "❌ Cloud Run deployment failed"
    exit 1
fi

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
echo "Service URL: $SERVICE_URL"

# Wait for service to be ready
echo ""
echo "Waiting for service to be ready..."
for i in {1..30}; do
    if curl -s "${SERVICE_URL}/health" | grep -q "healthy"; then
        echo "✅ Service health check passed"
        break
    fi
    echo "Attempt $i/30: Service not ready yet..."
    sleep 10
done

# Run post-deployment tests
echo ""
echo "Running post-deployment verification..."
python scripts/post_deployment_tests.py $SERVICE_URL

echo ""
echo "🎯 Production deployment completed successfully!"
echo "Service is live at: $SERVICE_URL"
```

### 9.2 Database Migration

**Checklist**:
- [ ] Migration scripts tested
- [ ] Backup created before migration
- [ ] Migration executed successfully
- [ ] Data integrity verified
- [ ] Rollback tested

## Phase 10: Post-Deployment Verification

### 10.1 Production Health Checks

**Checklist**:
- [ ] All endpoints responding
- [ ] Database connectivity verified
- [ ] External integrations working
- [ ] ML models loading correctly
- [ ] Authentication system functional

**Post-Deployment Test Script**:
```python
# scripts/post_deployment_tests.py
import sys
import requests
import asyncio
import json
from typing import Dict, Any

class PostDeploymentTester:
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 30
    
    def test_health_endpoint(self) -> bool:
        """Test basic health endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                assert data.get("status") == "healthy"
                print("✅ Health endpoint working")
                return True
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health endpoint error: {e}")
            return False
    
    def test_api_endpoints(self) -> bool:
        """Test critical API endpoints"""
        endpoints = [
            "/api/v1/health-data/upload",
            "/api/v1/auth/verify",
            "/api/v1/ml/pat/status"
        ]
        
        for endpoint in endpoints:
            try:
                response = self.session.options(f"{self.base_url}{endpoint}")
                if response.status_code in [200, 204, 405]:  # OPTIONS may return 405
                    print(f"✅ Endpoint {endpoint} accessible")
                else:
                    print(f"❌ Endpoint {endpoint} failed: {response.status_code}")
                    return False
            except Exception as e:
                print(f"❌ Endpoint {endpoint} error: {e}")
                return False
        
        return True
    
    def test_database_connectivity(self) -> bool:
        """Test database connectivity through API"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/system/db-status")
            if response.status_code == 200:
                data = response.json()
                if data.get("database_status") == "connected":
                    print("✅ Database connectivity verified")
                    return True
                else:
                    print("❌ Database not connected")
                    return False
            else:
                print(f"❌ Database status check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Database connectivity error: {e}")
            return False
    
    def test_ml_model_status(self) -> bool:
        """Test ML model loading status"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/ml/status")
            if response.status_code == 200:
                data = response.json()
                models_status = data.get("models", {})
                
                if models_status.get("pat_model") == "loaded":
                    print("✅ PAT model loaded successfully")
                    return True
                else:
                    print("❌ PAT model not loaded")
                    return False
            else:
                print(f"❌ ML model status check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ ML model status error: {e}")
            return False
    
    def test_authentication_system(self) -> bool:
        """Test authentication system"""
        try:
            # Test without authentication (should fail)
            response = self.session.get(f"{self.base_url}/api/v1/protected/test")
            if response.status_code == 401:
                print("✅ Authentication protection working")
                return True
            else:
                print(f"❌ Authentication system issue: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Authentication test error: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all post-deployment tests"""
        print("🧪 Post-Deployment Verification")
        print("==============================")
        
        tests = [
            self.test_health_endpoint,
            self.test_api_endpoints,
            self.test_database_connectivity,
            self.test_ml_model_status,
            self.test_authentication_system
        ]
        
        success = True
        for test in tests:
            try:
                result = test()
                success &= result
            except Exception as e:
                print(f"❌ Test {test.__name__} failed with exception: {e}")
                success = False
        
        return success

def main():
    if len(sys.argv) != 2:
        print("Usage: python post_deployment_tests.py <service_url>")
        sys.exit(1)
    
    service_url = sys.argv[1]
    tester = PostDeploymentTester(service_url)
    
    success = tester.run_all_tests()
    
    if success:
        print("\n🎯 All post-deployment tests passed!")
        print("🌟 Production deployment verified successfully!")
        sys.exit(0)
    else:
        print("\n💥 Some post-deployment tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 10.2 Performance Monitoring

**Checklist**:
- [ ] Response time monitoring active
- [ ] Error rate monitoring configured
- [ ] Resource utilization tracking
- [ ] Log aggregation working
- [ ] Alert rules configured

## Build Automation & CI/CD

### Complete Build Pipeline

**GitHub Actions Workflow**:
```yaml
# .github/workflows/build-and-deploy.yml
name: Build and Deploy

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  SERVICE_NAME: clarity-backend
  REGION: us-central1

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install UV
      run: pip install uv
    
    - name: Install dependencies
      run: uv pip install -r requirements.txt -r requirements-dev.txt
    
    - name: Run environment verification
      run: ./scripts/verify-environment.sh
    
    - name: Run security scan
      run: ./scripts/security-scan.sh
    
    - name: Run quality checks
      run: ./scripts/quality-check.sh
    
    - name: Run unit tests
      run: ./scripts/run-unit-tests.sh
    
    - name: Run integration tests
      run: ./scripts/run-integration-tests.sh
    
    - name: Run HIPAA compliance tests
      run: ./scripts/hipaa-compliance-check.sh
    
    - name: Test ML models
      run: python scripts/test_ml_models.py
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Google Cloud CLI
      uses: google-github-actions/setup-gcloud@v1
      with:
        service_account_key: ${{ secrets.GCP_SA_KEY }}
        project_id: ${{ secrets.GCP_PROJECT_ID }}
    
    - name: Configure Docker for GCR
      run: gcloud auth configure-docker
    
    - name: Build Docker image
      run: ./scripts/build-docker.sh
    
    - name: Push to Container Registry
      run: |
        docker tag clarity-backend:latest gcr.io/$PROJECT_ID/clarity-backend:$GITHUB_SHA
        docker push gcr.io/$PROJECT_ID/clarity-backend:$GITHUB_SHA

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Google Cloud CLI
      uses: google-github-actions/setup-gcloud@v1
      with:
        service_account_key: ${{ secrets.GCP_SA_KEY }}
        project_id: ${{ secrets.GCP_PROJECT_ID }}
    
    - name: Run deployment readiness check
      run: ./scripts/deployment-readiness.sh
    
    - name: Deploy to production
      run: ./scripts/deploy-production.sh $GITHUB_SHA
    
    - name: Run post-deployment tests
      run: |
        SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
        python scripts/post_deployment_tests.py $SERVICE_URL
```

## Build Metrics and Monitoring

### Build Success Metrics

**Key Performance Indicators**:
- **Build Success Rate**: Target ≥95%
- **Build Duration**: Target <60 minutes
- **Test Coverage**: Target ≥90%
- **Security Issues**: Target 0 high/critical
- **Deployment Success Rate**: Target ≥98%

### Monitoring Dashboard

**Build Health Dashboard**:
```python
# scripts/build_metrics_dashboard.py
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

class BuildMetrics:
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_build_success_rate(self, days: int = 30) -> float:
        """Calculate build success rate over specified period"""
        # Implementation would query CI/CD logs
        # Mock data for example
        total_builds = 150
        successful_builds = 143
        return (successful_builds / total_builds) * 100
    
    def get_average_build_time(self, days: int = 30) -> float:
        """Get average build time in minutes"""
        # Implementation would analyze build logs
        return 42.5  # minutes
    
    def get_test_coverage_trend(self, days: int = 30) -> List[Dict]:
        """Get test coverage trend over time"""
        # Mock trend data
        return [
            {"date": "2024-01-01", "coverage": 89.2},
            {"date": "2024-01-02", "coverage": 90.1},
            {"date": "2024-01-03", "coverage": 91.5},
        ]
    
    def get_security_issues_count(self, days: int = 30) -> Dict[str, int]:
        """Get security issues count by severity"""
        return {
            "critical": 0,
            "high": 0,
            "medium": 2,
            "low": 5
        }
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate complete dashboard data"""
        return {
            "build_success_rate": self.calculate_build_success_rate(),
            "average_build_time": self.get_average_build_time(),
            "test_coverage_trend": self.get_test_coverage_trend(),
            "security_issues": self.get_security_issues_count(),
            "last_updated": datetime.now().isoformat()
        }

def main():
    metrics = BuildMetrics()
    dashboard_data = metrics.generate_dashboard_data()
    
    print("📊 Build Metrics Dashboard")
    print("==========================")
    print(f"Build Success Rate: {dashboard_data['build_success_rate']:.1f}%")
    print(f"Average Build Time: {dashboard_data['average_build_time']:.1f} minutes")
    print(f"Security Issues: {dashboard_data['security_issues']}")
    print(f"Latest Coverage: {dashboard_data['test_coverage_trend'][-1]['coverage']}%")

if __name__ == "__main__":
    main()
```

## Emergency Procedures

### Build Failure Response

**Immediate Actions**:
1. **Stop deployment pipeline**
2. **Notify development team**
3. **Preserve build artifacts for analysis**
4. **Initiate rollback if production affected**
5. **Document incident details**

**Build Failure Analysis Script**:
```bash
#!/bin/bash
# scripts/build-failure-analysis.sh

echo "🚨 Build Failure Analysis"
echo "========================"

BUILD_ID=${1:-latest}
echo "Analyzing build: $BUILD_ID"

# Collect build logs
echo "Collecting build logs..."
mkdir -p build-analysis/$BUILD_ID

# Copy test results
cp -r htmlcov/ build-analysis/$BUILD_ID/ 2>/dev/null || echo "No coverage reports found"
cp pytest-results.xml build-analysis/$BUILD_ID/ 2>/dev/null || echo "No test results found"
cp bandit-report.json build-analysis/$BUILD_ID/ 2>/dev/null || echo "No security scan results found"

# Generate failure summary
echo "Generating failure summary..."
python scripts/analyze_build_failure.py $BUILD_ID > build-analysis/$BUILD_ID/failure-summary.txt

echo "Build failure analysis saved to: build-analysis/$BUILD_ID/"
echo "Review logs and take corrective action."
```

### Rollback Procedures

**Rollback Checklist**:
- [ ] Previous version identified
- [ ] Database rollback plan ready
- [ ] Rollback executed
- [ ] Health checks verified
- [ ] Incident documentation updated

Remember: **Every build step must be automated, verifiable, and auditable for healthcare compliance.**
