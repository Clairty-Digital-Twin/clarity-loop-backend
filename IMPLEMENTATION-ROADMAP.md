# Clarity Loop Backend - Vertical Slice Implementation Roadmap

## 🎯 Objective
Build a production-grade HealthKit wellness platform backend using **vertical slice development** with bulletproof testing at each step. Each slice is fully functional from API endpoint → business logic → data storage → testing.

## 🛡️ Core Principles

1. **Vertical Slice Development** - Complete functionality from API to storage for each feature
2. **Quality Gates at Every Step** - MyPy, Ruff, PyTest, Bandit, and curl testing  
3. **Error-Driven Development** - Use tools to guide implementation, fix one error type at a time
4. **HIPAA/Security First** - Security scanning and privacy compliance at every step
5. **Documentation Fidelity** - Implementation must match documented contracts exactly

## 📁 Required Directory Structure

```
src/clarity/
├── __init__.py
├── main.py                 # FastAPI app entry point
├── config/
│   ├── __init__.py
│   ├── settings.py         # Environment configuration
│   └── firebase_config.py  # Firebase initialization
├── auth/
│   ├── __init__.py
│   ├── middleware.py       # Firebase JWT verification
│   └── dependencies.py    # FastAPI dependency injection
├── api/
│   ├── __init__.py
│   ├── health/
│   │   ├── __init__.py
│   │   ├── routes.py       # Health data endpoints
│   │   └── models.py       # Pydantic request/response models
│   └── user/
│       ├── __init__.py
│       ├── routes.py       # User management endpoints  
│       └── models.py       # User-related models
├── services/
│   ├── __init__.py
│   ├── firestore_service.py    # Firestore operations
│   ├── healthkit_service.py    # HealthKit data processing
│   └── validation_service.py   # Data validation logic
└── utils/
    ├── __init__.py
    ├── logging.py          # HIPAA-compliant logging
    └── exceptions.py       # Custom exception classes

tests/
├── __init__.py
├── conftest.py             # PyTest fixtures and configuration
├── unit/
│   ├── test_auth/
│   ├── test_services/
│   └── test_utils/
├── integration/
│   ├── test_health_endpoints.py
│   ├── test_user_endpoints.py
│   └── test_end_to_end.py
└── fixtures/
    ├── healthkit_samples.json      # Sample HealthKit payloads
    ├── firebase_tokens.json       # Mock Firebase tokens
    └── test_users.json            # Test user data
```

## 🔧 Quality Gate Commands

Run after **every phase** to prevent technical debt:

```bash
# 1. Type checking (zero tolerance for errors)
mypy src/clarity/ --strict

# 2. Code formatting and linting
ruff format src/clarity/
ruff check src/clarity/ --fix

# 3. Security scanning
bandit -r src/clarity/ -f json

# 4. Testing with coverage
pytest tests/ --cov=src/clarity --cov-report=term-missing --cov-fail-under=90

# 5. Import sorting
isort src/clarity/ tests/

# 6. Integration testing
python scripts/test_endpoints.py
```

## 🚀 VERTICAL SLICE 1: Health Data Upload (MVP)

**Goal**: Complete HealthKit data ingestion pipeline with authentication, validation, and storage.

### Phase 1A: Foundation Setup (30 minutes)

**Deliverables**:
1. Create directory structure
2. Basic FastAPI app with health check endpoint
3. Environment configuration (Firebase project ID, Firestore DB)
4. HIPAA-compliant logging setup
5. Basic error handling

**Test Criteria**:
```bash
# Start server
uvicorn src.clarity.main:app --reload

# Test health check
curl http://localhost:8000/health
# Expected: {"status": "healthy", "timestamp": "2025-06-01T12:36:24Z"}
```

**Quality Gate**: MyPy + Ruff + PyTest (basic structure tests)

### Phase 1B: Firebase Authentication (45 minutes)

**Deliverables**:
1. Firebase JWT verification middleware
2. Authenticated dependency injection for protected routes
3. Unit tests with mock Firebase tokens
4. Error handling for invalid/expired tokens

**Test Criteria**:
```bash
# Test with valid token
curl -X GET http://localhost:8000/v1/health/protected \
  -H "Authorization: Bearer $FIREBASE_TOKEN"
# Expected: 200 OK with user info

# Test with invalid token
curl -X GET http://localhost:8000/v1/health/protected \
  -H "Authorization: Bearer invalid_token"
# Expected: 401 Unauthorized
```

**Quality Gate**: MyPy + Ruff + PyTest + Security scan

### Phase 1C: HealthKit Upload Models (30 minutes)

**Deliverables**:
1. Pydantic models matching HealthKit Upload v1 contract
2. Request/response schemas with proper validation
3. Unit tests for model validation
4. Error response models

**Test Criteria**:
```python
# Test model validation
from src.clarity.api.health.models import HealthKitUploadRequest
payload = {...}  # Valid HealthKit payload
request = HealthKitUploadRequest(**payload)
assert request.device_time_zone == "America/Los_Angeles"
```

**Quality Gate**: MyPy + Ruff + PyTest

### Phase 1D: HealthKit Upload Endpoint (60 minutes)

**Deliverables**:
1. POST /v1/health/data/upload route handler
2. Comprehensive validation logic (UUID dedup, time ordering, unit conversion)
3. Error handling with proper HTTP status codes
4. Integration tests with realistic HealthKit payloads

**Test Criteria**:
```bash
# Test successful upload
curl -X POST http://localhost:8000/v1/health/data/upload \
  -H "Authorization: Bearer $FIREBASE_TOKEN" \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/healthkit_samples.json
# Expected: 201 Created with upload confirmation

# Test validation errors
curl -X POST http://localhost:8000/v1/health/data/upload \
  -H "Authorization: Bearer $FIREBASE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"invalid": "payload"}'
# Expected: 422 Unprocessable Entity with validation errors
```

**Quality Gate**: MyPy + Ruff + PyTest + Security scan

### Phase 1E: Firestore Integration (45 minutes)

**Deliverables**:
1. Firestore client configuration with proper error handling
2. Data persistence service with encryption at rest
3. UploadId idempotency handling
4. Integration tests with Firestore emulator

**Test Criteria**:
```bash
# Start Firestore emulator
firebase emulators:start --only firestore

# Test end-to-end upload and storage
curl -X POST http://localhost:8000/v1/health/data/upload \
  -H "Authorization: Bearer $FIREBASE_TOKEN" \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/healthkit_samples.json

# Verify data in Firestore
# Check uploadId idempotency (duplicate upload should not double-store)
```

**Quality Gate**: MyPy + Ruff + PyTest + Security scan

### Phase 1F: End-to-End Validation (30 minutes)

**Deliverables**:
1. Comprehensive end-to-end test scripts
2. Performance benchmarks
3. Documentation of curl commands for manual testing
4. Deployment validation

**Test Criteria**:
```bash
# Complete workflow test
python scripts/test_healthkit_upload_e2e.py
# Expected: All tests pass with performance metrics
```

**Quality Gate**: All checks + performance benchmarks + security audit

**Total Time for VERTICAL SLICE 1**: ~4 hours

## 🔄 VERTICAL SLICE 2: User Management (2-3 hours)

**Goal**: Complete user profile management and Apple privacy compliance.

**Key Endpoints**:
- GET /v1/user/profile
- POST /v1/user/profile  
- DELETE /v1/user/{uid}/purge (Apple privacy requirement)

## 🤖 VERTICAL SLICE 3: ML Processing Pipeline (4-5 hours)

**Goal**: Pub/Sub integration and PAT model inference.

**Key Components**:
- Pub/Sub message handling for async processing
- PAT model inference integration
- Basic health insights generation

## 📊 VERTICAL SLICE 4: Advanced Analytics (6-8 hours)

**Goal**: Gemini LLM integration and advanced insights.

**Key Components**:
- Gemini LLM integration with quota handling
- Complex insights generation
- Historical trend analysis

## ⚠️ Risk Mitigation Strategies

1. **Complexity Creep Risk**: Strict vertical slice discipline
2. **Technical Debt**: Zero tolerance for type errors, >90% test coverage
3. **Security Drift**: Security scan in every quality gate
4. **Integration Gaps**: Comprehensive curl scripts for validation
5. **Documentation Drift**: Update docs when implementation differs

## 🚦 Success Criteria for Each Slice

1. ✅ **Functional**: All curl commands work end-to-end
2. ✅ **Tested**: >90% test coverage, all quality gates pass
3. ✅ **Secure**: Bandit scan clean, HIPAA compliant
4. ✅ **Documented**: Implementation matches API documentation
5. ✅ **Deployable**: Can deploy to Cloud Run and test remotely

## 🎬 Next Action

**START IMMEDIATELY** with VERTICAL SLICE 1, Phase 1A (Foundation Setup):

1. Create `src/clarity/main.py`
2. Implement basic FastAPI app with health check
3. Test with curl
4. Run quality gates
5. Proceed to Phase 1B

Ready to build production-grade backend without complexity spirals! 🚀
