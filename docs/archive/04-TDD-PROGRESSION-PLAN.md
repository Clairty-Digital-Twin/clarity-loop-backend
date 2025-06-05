# Test-Driven Development Progression Plan

**Canonical approach for incremental TDD adoption in healthcare AI systems**

## Core Philosophy

**Progressive TDD Integration**: Start where you are, incrementally adopt TDD practices without disrupting development velocity. Build confidence through small wins before tackling complex healthcare AI scenarios.

**Domain-Driven Test Design**: Healthcare systems have unique testing requirements - from HIPAA compliance to ML model validation. Tests must reflect real clinical workflows and patient safety requirements.

**Risk-Stratified Testing**: Critical healthcare features (authentication, data security, ML predictions) require different TDD rigor than utility functions.

## TDD Maturity Levels

### Level 0: Test-After Development (Current State)

**Characteristics**: Write code first, add tests later, reactive testing approach

**Typical Workflow**:

```python
# 1. Write implementation first
@router.post("/health-data/upload")
async def upload_health_data(data: dict):
    # Implementation code here
    return {"status": "uploaded"}

# 2. Add tests afterward
def test_upload_health_data():
    # Test written after implementation
    assert True  # Basic test
```

### Level 1: Test-First for New Features

**Goal**: Write tests before implementation for all new features
**Timeline**: 2-4 weeks to establish habit
**Success Criteria**: 80% of new functions have tests written first

**Implementation Strategy**:

```python
# 1. Write failing test first
def test_heart_rate_validation():
    """Test heart rate data validation"""
    with pytest.raises(ValidationError):
        HeartRateData(bpm=300)  # Invalid heart rate

    # This test will fail initially
    valid_data = HeartRateData(bpm=72)
    assert valid_data.bpm == 72

# 2. Write minimal implementation to pass
class HeartRateData(BaseModel):
    bpm: int

    @validator('bpm')
    def validate_bpm(cls, v):
        if v < 30 or v > 250:  # Reasonable human heart rate range
            raise ValueError('Invalid heart rate')
        return v

# 3. Refactor with confidence
class HeartRateData(BaseModel):
    bpm: int
    timestamp: datetime
    source: str = "apple_watch"

    @validator('bpm')
    def validate_bpm(cls, v):
        if not 30 <= v <= 250:
            raise ValueError(f'Heart rate {v} outside valid range (30-250)')
        return v
```

### Level 2: Red-Green-Refactor Discipline

**Goal**: Consistent TDD cycle for all development
**Timeline**: 4-6 weeks to internalize rhythm
**Success Criteria**: Can demonstrate complete TDD cycles in code reviews

**The Healthcare TDD Cycle**:

```python
# RED: Write failing test that captures healthcare requirement
async def test_patient_data_encryption_at_rest():
    """Ensure patient data is encrypted when stored (HIPAA requirement)"""
    patient_data = PatientHealthData(
        patient_id="patient_123",
        heart_rate=72,
        timestamp=datetime.utcnow()
    )

    # Store the data
    doc_id = await health_repository.store(patient_data)

    # Retrieve raw storage to verify encryption
    raw_doc = await firestore_client.collection("health_data").document(doc_id).get()
    raw_data = raw_doc.to_dict()

    # HIPAA requires encryption - this should NOT be readable
    assert "patient_123" not in str(raw_data)  # Should fail initially
    assert "heart_rate" not in str(raw_data)  # Should fail initially

# GREEN: Minimal implementation to pass
class HealthDataRepository:
    async def store(self, patient_data: PatientHealthData) -> str:
        # Encrypt sensitive data before storage
        encrypted_data = {
            "encrypted_payload": encrypt_patient_data(patient_data.dict()),
            "created_at": datetime.utcnow().isoformat()
        }

        doc_ref = await self.firestore.collection("health_data").add(encrypted_data)
        return doc_ref.id

# REFACTOR: Improve design while keeping tests green
class HealthDataRepository:
    def __init__(self, encryption_service: EncryptionService):
        self.encryption = encryption_service

    async def store(self, patient_data: PatientHealthData) -> str:
        # Structured approach to encryption
        sensitive_fields = patient_data.get_sensitive_fields()
        encrypted_payload = await self.encryption.encrypt_structured(
            data=patient_data.dict(),
            sensitive_fields=sensitive_fields
        )

        document = {
            "encrypted_payload": encrypted_payload,
            "metadata": {
                "data_type": patient_data.data_type,
                "source": patient_data.source,
                "created_at": datetime.utcnow().isoformat(),
                "encryption_version": "v2"
            }
        }

        doc_ref = await self.firestore.collection("health_data").add(document)
        return doc_ref.id
```

### Level 3: Outside-In TDD

**Goal**: Start with acceptance tests, drive implementation from user needs
**Timeline**: 6-8 weeks to master technique
**Success Criteria**: Can build complete features starting from user stories

**Outside-In Healthcare Example**:

```python
# 1. Start with acceptance test (outside)
@pytest.mark.integration
async def test_complete_health_data_workflow():
    """
    User Story: As a patient, I want to upload heart rate data from my Apple Watch
    so that my doctor can monitor my cardiac health trends.
    """
    # Arrange: Simulate Apple Watch data
    apple_watch_data = {
        "data_type": "heart_rate",
        "readings": [
            {"timestamp": "2024-01-01T12:00:00Z", "bpm": 72},
            {"timestamp": "2024-01-01T12:01:00Z", "bpm": 75},
        ],
        "device_id": "apple_watch_series_9",
        "user_id": "patient_123"
    }

    # Act: Upload data through API
    response = await authenticated_client.post(
        "/api/v1/health-data/upload",
        json=apple_watch_data
    )

    # Assert: Verify complete workflow
    assert response.status_code == 202
    upload_id = response.json()["upload_id"]

    # Verify data is stored securely
    stored_data = await health_repository.get_upload(upload_id)
    assert stored_data.user_id == "patient_123"
    assert len(stored_data.readings) == 2

    # Verify AI processing is triggered
    processing_job = await get_processing_job(upload_id)
    assert processing_job.status == "queued"

    # Verify audit trail for HIPAA compliance
    audit_logs = await get_audit_logs("patient_123", "data_upload")
    assert len(audit_logs) > 0
    assert audit_logs[-1]["action"] == "health_data_upload"

# 2. Write unit tests for each component (inside)
class TestHealthDataUploadAPI:
    async def test_validates_required_fields(self):
        # This test will drive the validation logic
        pass

    async def test_handles_authentication(self):
        # This test will drive the auth integration
        pass

class TestHealthDataRepository:
    async def test_encrypts_patient_data(self):
        # This test will drive the encryption implementation
        pass

class TestHealthDataProcessor:
    async def test_triggers_ml_analysis(self):
        # This test will drive the ML pipeline integration
        pass
```

### Level 4: Advanced TDD Patterns

**Goal**: Master complex TDD scenarios for healthcare AI
**Timeline**: 8-12 weeks for full proficiency
**Success Criteria**: Can TDD complex ML pipelines and real-time systems

## TDD Implementation Strategy by Domain

### 1. API Endpoints (High Risk - Full TDD)

#### Authentication & Authorization

```python
# Test-first approach for security-critical features
class TestAuthenticationTDD:

    def test_jwt_token_validation_rejects_expired_tokens(self):
        """Security requirement: Expired tokens must be rejected"""
        # RED: This test will fail until we implement token validation
        expired_token = create_expired_jwt_token()

        with pytest.raises(TokenExpiredError):
            validate_jwt_token(expired_token)

    def test_jwt_token_validation_rejects_malformed_tokens(self):
        """Security requirement: Malformed tokens must be rejected"""
        malformed_token = "not.a.valid.jwt"

        with pytest.raises(InvalidTokenError):
            validate_jwt_token(malformed_token)

    def test_user_permissions_prevent_cross_patient_access(self):
        """HIPAA requirement: Users can only access their own data"""
        user_token = create_valid_token(user_id="patient_123")

        # Should allow access to own data
        assert can_access_patient_data(user_token, "patient_123") == True

        # Should deny access to other patient data
        assert can_access_patient_data(user_token, "patient_456") == False

# Implementation driven by tests
def validate_jwt_token(token: str) -> TokenPayload:
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=["HS256"],
            options={"verify_exp": True}  # Driven by expired token test
        )
        return TokenPayload(**payload)
    except jwt.ExpiredSignatureError:
        raise TokenExpiredError("Token has expired")  # Driven by test
    except jwt.InvalidTokenError:
        raise InvalidTokenError("Invalid token format")  # Driven by test
```

#### Data Validation

```python
class TestHealthDataValidationTDD:

    def test_heart_rate_validation_accepts_normal_range(self):
        """Clinical requirement: Normal heart rates 60-100 bpm"""
        # GREEN path test
        for bpm in [60, 72, 80, 100]:
            data = HeartRateReading(bpm=bpm)
            assert data.bpm == bpm

    def test_heart_rate_validation_flags_bradycardia(self):
        """Clinical requirement: Heart rate < 60 should be flagged"""
        reading = HeartRateReading(bpm=45)

        assert reading.is_bradycardia == True
        assert "bradycardia" in reading.clinical_flags

    def test_heart_rate_validation_flags_tachycardia(self):
        """Clinical requirement: Heart rate > 100 should be flagged"""
        reading = HeartRateReading(bpm=120)

        assert reading.is_tachycardia == True
        assert "tachycardia" in reading.clinical_flags

    def test_heart_rate_validation_rejects_impossible_values(self):
        """Safety requirement: Reject physiologically impossible values"""
        with pytest.raises(ValidationError, match="physiologically impossible"):
            HeartRateReading(bpm=500)

        with pytest.raises(ValidationError, match="physiologically impossible"):
            HeartRateReading(bpm=0)

# Implementation emerges from tests
class HeartRateReading(BaseModel):
    bpm: int
    timestamp: datetime
    clinical_flags: List[str] = Field(default_factory=list)

    @validator('bpm')
    def validate_bpm_range(cls, v):
        if v < 20 or v > 300:  # Driven by impossible values test
            raise ValueError("Heart rate physiologically impossible")
        return v

    @property
    def is_bradycardia(self) -> bool:
        return self.bpm < 60  # Driven by bradycardia test

    @property
    def is_tachycardia(self) -> bool:
        return self.bpm > 100  # Driven by tachycardia test

    def __post_init__(self):
        # Driven by clinical flags tests
        if self.is_bradycardia:
            self.clinical_flags.append("bradycardia")
        if self.is_tachycardia:
            self.clinical_flags.append("tachycardia")
```

### 2. ML Model Integration (Medium Risk - Structured TDD)

#### PAT Model Testing

```python
class TestPATModelTDD:

    def test_actigraphy_preprocessing_normalizes_data(self):
        """ML requirement: Data must be z-score normalized"""
        raw_data = np.array([1, 0, 1, 1, 0, 0, 1] * 200)  # 24h of data

        processed = preprocess_actigraphy_for_pat(raw_data)

        # Should be z-score normalized
        assert abs(processed.mean()) < 0.1  # Approximately zero mean
        assert abs(processed.std() - 1.0) < 0.1  # Approximately unit variance

    def test_pat_model_returns_valid_depression_probability(self):
        """ML requirement: Model output must be valid probability"""
        sample_data = generate_sample_actigraphy_data()

        prediction = pat_model.predict_depression_risk(sample_data)

        assert isinstance(prediction, float)
        assert 0.0 <= prediction <= 1.0  # Valid probability range

    def test_pat_model_handles_missing_data_gracefully(self):
        """Robustness requirement: Handle incomplete data"""
        incomplete_data = generate_incomplete_actigraphy_data()  # Missing hours

        # Should not crash
        prediction = pat_model.predict_depression_risk(incomplete_data)

        # Should return valid prediction or None
        assert prediction is None or (0.0 <= prediction <= 1.0)

    def test_pat_model_prediction_consistency(self):
        """Reliability requirement: Same input -> same output"""
        test_data = generate_deterministic_actigraphy_data()

        prediction1 = pat_model.predict_depression_risk(test_data)
        prediction2 = pat_model.predict_depression_risk(test_data)

        assert abs(prediction1 - prediction2) < 1e-6  # Deterministic

# Implementation driven by tests
class PATModel:
    def __init__(self, model_path: str):
        self.model = torch.load(model_path)
        self.model.eval()

    def predict_depression_risk(self, actigraphy_data: np.ndarray) -> Optional[float]:
        try:
            # Driven by missing data test
            if self._has_insufficient_data(actigraphy_data):
                return None

            # Driven by preprocessing test
            processed = preprocess_actigraphy_for_pat(actigraphy_data)

            # Driven by consistency test
            with torch.no_grad():
                torch.manual_seed(42)  # Ensure deterministic output
                tensor_input = torch.FloatTensor(processed).unsqueeze(0)
                output = self.model(tensor_input)
                probability = torch.sigmoid(output).item()  # Driven by probability test

            return probability

        except Exception as e:
            logger.error(f"PAT model prediction failed: {e}")
            return None  # Driven by robustness test

def preprocess_actigraphy_for_pat(data: np.ndarray) -> np.ndarray:
    """Preprocessing driven by normalization test"""
    # Z-score normalization as required by test
    if data.std() == 0:
        return np.zeros_like(data)

    return (data - data.mean()) / data.std()
```

### 3. Database Operations (Medium Risk - Contract Testing)

```python
class TestHealthDataRepositoryTDD:

    async def test_store_encrypts_sensitive_patient_data(self):
        """HIPAA requirement: PII must be encrypted at rest"""
        patient_data = PatientHealthData(
            patient_id="patient_123",
            ssn="123-45-6789",  # Sensitive PII
            heart_rate=72
        )

        doc_id = await repository.store(patient_data)

        # Verify encryption by checking raw storage
        raw_doc = await firestore_client.collection("health_data").document(doc_id).get()
        raw_content = str(raw_doc.to_dict())

        # PII should not appear in plaintext
        assert "patient_123" not in raw_content
        assert "123-45-6789" not in raw_content

    async def test_retrieve_decrypts_patient_data_correctly(self):
        """Functional requirement: Encrypted data must be retrievable"""
        original_data = PatientHealthData(
            patient_id="patient_456",
            heart_rate=85
        )

        # Store then retrieve
        doc_id = await repository.store(original_data)
        retrieved_data = await repository.retrieve(doc_id)

        # Should match original
        assert retrieved_data.patient_id == "patient_456"
        assert retrieved_data.heart_rate == 85

    async def test_audit_logging_records_all_data_access(self):
        """HIPAA requirement: All data access must be audited"""
        patient_id = "patient_789"

        # Access patient data
        await repository.get_patient_data(patient_id)

        # Verify audit log entry
        audit_logs = await audit_service.get_logs(patient_id)

        assert len(audit_logs) > 0
        latest_log = audit_logs[-1]
        assert latest_log["action"] == "data_access"
        assert latest_log["patient_id"] == patient_id
        assert latest_log["timestamp"] is not None

# Implementation contracts driven by tests
class HealthDataRepository:
    def __init__(self, firestore_client, encryption_service, audit_service):
        self.firestore = firestore_client
        self.encryption = encryption_service
        self.audit = audit_service

    async def store(self, patient_data: PatientHealthData) -> str:
        # Driven by encryption test
        encrypted_payload = await self.encryption.encrypt(
            patient_data.dict()
        )

        document = {
            "encrypted_data": encrypted_payload,
            "created_at": datetime.utcnow().isoformat()
        }

        doc_ref = await self.firestore.collection("health_data").add(document)

        # Driven by audit test
        await self.audit.log_action(
            action="data_store",
            patient_id=patient_data.patient_id,
            document_id=doc_ref.id
        )

        return doc_ref.id

    async def retrieve(self, doc_id: str) -> PatientHealthData:
        doc = await self.firestore.collection("health_data").document(doc_id).get()

        if not doc.exists:
            raise DocumentNotFoundError(f"Document {doc_id} not found")

        # Driven by decryption test
        encrypted_data = doc.to_dict()["encrypted_data"]
        decrypted_data = await self.encryption.decrypt(encrypted_data)

        return PatientHealthData(**decrypted_data)
```

## TDD Adoption Timeline

### Weeks 1-2: Foundation Building

**Focus**: Establish basic TDD habits for new features

**Daily Practice**:

- Write one test before implementation each day
- Practice Red-Green-Refactor on utility functions
- Set up testing infrastructure and CI integration

**Deliverables**:

- [ ] All new utility functions are test-driven
- [ ] Team members comfortable with pytest and basic TDD
- [ ] CI pipeline runs tests on every commit

### Weeks 3-4: API Endpoint TDD

**Focus**: Apply TDD to all new API endpoints

**Practice Areas**:

- Request/response validation
- Authentication and authorization
- Error handling and edge cases

**Deliverables**:

- [ ] All new API endpoints written test-first
- [ ] Comprehensive test coverage for authentication
- [ ] Error scenarios thoroughly tested

### Weeks 5-6: Business Logic TDD

**Focus**: Test-drive domain logic and services

**Practice Areas**:

- Health data processing logic
- Clinical decision rules
- Business rule validation

**Deliverables**:

- [ ] Health data validation rules are test-driven
- [ ] Clinical algorithms have comprehensive test coverage
- [ ] Business logic is decoupled and testable

### Weeks 7-8: Integration and ML TDD

**Focus**: Test-drive complex integrations and ML components

**Practice Areas**:

- Database operations and repositories
- External service integrations
- ML model wrapping and validation

**Deliverables**:

- [ ] Database operations are contract-tested
- [ ] ML model integration is test-driven
- [ ] External service calls are properly mocked/stubbed

### Weeks 9-12: Advanced TDD Mastery

**Focus**: Master outside-in TDD and complex scenarios

**Practice Areas**:

- Full user story implementation from acceptance tests
- Performance and load testing integration
- HIPAA compliance verification through tests

**Deliverables**:

- [ ] Complete features built outside-in from user stories
- [ ] Performance requirements verified through tests
- [ ] HIPAA compliance requirements tested automatically

## TDD Tools and Infrastructure

### Testing Framework Configuration

```toml
# pyproject.toml - TDD-optimized pytest configuration
[tool.pytest.ini_options]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--tb=short",
    "--cov=clarity",
    "--cov-report=term-missing",
    "--cov-report=html:htmlcov",
    "--cov-fail-under=90",
    "-ra",  # Show all test results except passed
    "--durations=10",  # Show slowest 10 tests
]

# TDD-specific markers
markers = [
    "unit: Unit tests for TDD",
    "integration: Integration tests for outside-in TDD",
    "acceptance: Acceptance tests for user stories",
    "contract: Contract tests for external dependencies",
    "performance: Performance tests for non-functional requirements",
    "security: Security tests for HIPAA compliance",
    "ml: Machine learning model tests",
]

# TDD-friendly test discovery
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*", "*Test", "*Tests"]
python_functions = ["test_*"]

# Async testing support for healthcare APIs
asyncio_mode = "auto"
```

### TDD Development Scripts

```bash
#!/bin/bash
# scripts/tdd-cycle.sh - Automated TDD workflow

echo "🔴 RED: Running tests (expecting failures)..."
pytest -x --tb=line --no-cov -q

if [ $? -eq 0 ]; then
    echo "⚠️  All tests passing - write a failing test first!"
    exit 1
fi

echo ""
echo "✅ Good! Tests are failing. Now implement code to make them pass."
echo ""
echo "🟢 GREEN: When ready, run 'scripts/green-check.sh'"
```

```bash
#!/bin/bash
# scripts/green-check.sh - Verify GREEN phase

echo "🟢 GREEN: Checking if tests now pass..."
pytest -x --tb=short --no-cov

if [ $? -eq 0 ]; then
    echo "✅ Tests are passing! Ready for REFACTOR phase."
    echo "🔵 REFACTOR: Improve code while keeping tests green."
    echo "   Run 'scripts/refactor-check.sh' to verify refactoring."
else
    echo "❌ Tests still failing. Continue implementing."
    exit 1
fi
```

```bash
#!/bin/bash
# scripts/refactor-check.sh - Verify REFACTOR phase

echo "🔵 REFACTOR: Verifying tests remain green during refactoring..."

# Run full test suite with coverage
pytest --cov=clarity --cov-fail-under=90

if [ $? -eq 0 ]; then
    echo "✅ Refactoring successful! Tests remain green."
    echo "🎯 TDD cycle complete. Ready for next feature."
else
    echo "❌ Refactoring broke tests. Revert changes and try again."
    exit 1
fi
```

### IDE Integration for TDD

```json
// .vscode/settings.json - TDD-optimized VS Code setup
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "--tb=short",
        "--no-cov",
        "-v"
    ],
    "python.testing.autoTestDiscoverOnSaveEnabled": true,

    // TDD workflow shortcuts
    "python.testing.debugPort": 3000,

    // Auto-run tests on save for TDD
    "saveAndRun": {
        "commands": [
            {
                "match": "test_.*\\.py$",
                "cmd": "pytest ${file} -v",
                "useShortcut": false,
                "silent": false
            }
        ]
    },

    // TDD snippets
    "python.defaultInterpreterPath": "./venv/bin/python"
}
```

### Test Factories for Healthcare Data

```python
# tests/factories.py - Test data factories for TDD
import factory
from datetime import datetime, timedelta
from clarity.models import PatientHealthData, HeartRateReading

class PatientHealthDataFactory(factory.Factory):
    class Meta:
        model = PatientHealthData

    patient_id = factory.Sequence(lambda n: f"patient_{n:04d}")
    timestamp = factory.LazyFunction(datetime.utcnow)
    source = "apple_watch"

    @factory.post_generation
    def add_clinical_context(obj, create, extracted, **kwargs):
        """Add realistic clinical context for testing"""
        if extracted:
            obj.clinical_notes = extracted

class HeartRateReadingFactory(factory.Factory):
    class Meta:
        model = HeartRateReading

    bpm = factory.Faker('random_int', min=60, max=100)  # Normal range
    timestamp = factory.LazyFunction(datetime.utcnow)
    confidence = factory.Faker('random_element', elements=[0.95, 0.98, 0.99])

class BradycardiaReadingFactory(HeartRateReadingFactory):
    """Factory for testing bradycardia detection"""
    bpm = factory.Faker('random_int', min=30, max=59)

class TachycardiaReadingFactory(HeartRateReadingFactory):
    """Factory for testing tachycardia detection"""
    bpm = factory.Faker('random_int', min=101, max=180)

# Usage in TDD tests
def test_detect_bradycardia():
    # Arrange
    bradycardia_reading = BradycardiaReadingFactory()

    # Act
    analysis = analyze_heart_rate(bradycardia_reading)

    # Assert
    assert analysis.is_bradycardia == True
    assert "bradycardia" in analysis.clinical_flags
```

## TDD Success Metrics

### Code Quality Metrics

```python
# scripts/tdd_metrics.py - Track TDD adoption success
def calculate_tdd_metrics():
    """Calculate metrics to track TDD adoption success"""

    metrics = {
        "test_first_percentage": calculate_test_first_percentage(),
        "test_coverage": get_test_coverage(),
        "test_to_code_ratio": calculate_test_to_code_ratio(),
        "red_green_refactor_cycles": count_proper_tdd_cycles(),
        "defect_density": calculate_defect_density(),
        "time_to_feedback": measure_test_execution_time()
    }

    return metrics

def calculate_test_first_percentage():
    """Percentage of commits where tests were added before implementation"""
    # Analysis of git history to track test-first development
    pass

def calculate_test_to_code_ratio():
    """Ratio of test code lines to production code lines"""
    # Higher ratio indicates more thorough testing
    pass

def count_proper_tdd_cycles():
    """Number of proper Red-Green-Refactor cycles in recent development"""
    # Track commits that follow TDD pattern
    pass
```

### Healthcare-Specific TDD Metrics

```python
class HealthcareTDDMetrics:
    """TDD metrics specific to healthcare system requirements"""

    def hipaa_compliance_test_coverage(self):
        """Percentage of HIPAA requirements covered by tests"""
        hipaa_requirements = [
            "data_encryption_at_rest",
            "data_encryption_in_transit",
            "access_logging",
            "user_authentication",
            "authorization_controls",
            "audit_trail_completeness"
        ]

        covered_requirements = []
        for requirement in hipaa_requirements:
            if self.has_test_coverage(requirement):
                covered_requirements.append(requirement)

        return len(covered_requirements) / len(hipaa_requirements)

    def clinical_safety_test_coverage(self):
        """Coverage of clinical safety requirements"""
        safety_requirements = [
            "data_validation_bounds",
            "clinical_flag_detection",
            "emergency_value_alerts",
            "ml_model_confidence_thresholds"
        ]

        return self.calculate_requirement_coverage(safety_requirements)

    def ml_model_test_robustness(self):
        """Measure ML model testing comprehensiveness"""
        ml_test_categories = [
            "input_validation",
            "output_range_validation",
            "edge_case_handling",
            "performance_benchmarks",
            "bias_detection",
            "reproducibility"
        ]

        return self.calculate_ml_test_coverage(ml_test_categories)
```

## Common TDD Challenges and Solutions

### Challenge 1: Testing Complex ML Pipelines

**Problem**: ML models are hard to test deterministically

**TDD Solution**:

```python
# Instead of testing the ML model directly, test the interface
class TestMLModelInterface:

    def test_model_input_validation(self):
        """Test input validation without model complexity"""
        with pytest.raises(ValidationError):
            MLModelInterface.validate_input([])  # Empty input

        with pytest.raises(ValidationError):
            MLModelInterface.validate_input(None)  # None input

    def test_model_output_format(self):
        """Test output format consistency"""
        mock_model_response = {"prediction": 0.75, "confidence": 0.92}

        formatted_output = MLModelInterface.format_output(mock_model_response)

        assert "depression_risk" in formatted_output
        assert 0 <= formatted_output["depression_risk"] <= 1

    def test_model_error_handling(self):
        """Test graceful error handling"""
        with patch('model.predict') as mock_predict:
            mock_predict.side_effect = ModelInferenceError("GPU memory error")

            result = MLModelInterface.predict_safely(valid_input)

            assert result["status"] == "error"
            assert result["fallback_used"] == True
```

### Challenge 2: Testing Async Healthcare APIs

**Problem**: Complex async workflows are hard to test

**TDD Solution**:

```python
class TestAsyncHealthDataWorkflow:

    @pytest.mark.asyncio
    async def test_health_data_processing_pipeline(self):
        """Test complete async processing pipeline"""
        # Arrange
        health_data = create_test_health_data()

        # Act
        upload_result = await health_api.upload_data(health_data)

        # Assert - Test each stage
        assert upload_result["status"] == "accepted"

        # Verify async processing was triggered
        await asyncio.sleep(0.1)  # Allow async tasks to start

        processing_status = await health_api.get_processing_status(
            upload_result["upload_id"]
        )
        assert processing_status["status"] in ["processing", "completed"]

    @pytest.mark.asyncio
    async def test_concurrent_health_data_uploads(self):
        """Test system handles concurrent uploads correctly"""
        # Create multiple concurrent uploads
        upload_tasks = [
            health_api.upload_data(create_test_health_data())
            for _ in range(10)
        ]

        results = await asyncio.gather(*upload_tasks)

        # All uploads should succeed
        assert all(r["status"] == "accepted" for r in results)

        # All should have unique IDs
        upload_ids = [r["upload_id"] for r in results]
        assert len(set(upload_ids)) == len(upload_ids)
```

### Challenge 3: Testing HIPAA Compliance

**Problem**: Security requirements are abstract and hard to verify

**TDD Solution**:

```python
class TestHIPAAComplianceTDD:

    def test_data_encryption_requirement(self):
        """HIPAA 164.312(a)(2)(iv) - Encryption at rest"""
        patient_data = {"ssn": "123-45-6789", "name": "John Doe"}

        # Store data
        storage_service.store(patient_data)

        # Verify encryption by checking raw storage
        raw_storage = storage_service.get_raw_storage()

        # PII should not be readable in storage
        assert "123-45-6789" not in str(raw_storage)
        assert "John Doe" not in str(raw_storage)

    def test_access_logging_requirement(self):
        """HIPAA 164.312(b) - Audit controls"""
        patient_id = "patient_123"
        user_id = "doctor_456"

        # Access patient data
        with audit_context(user_id=user_id):
            patient_service.get_patient_data(patient_id)

        # Verify access was logged
        audit_logs = audit_service.get_logs(patient_id)

        assert len(audit_logs) > 0
        assert audit_logs[-1]["user_id"] == user_id
        assert audit_logs[-1]["action"] == "data_access"
        assert audit_logs[-1]["timestamp"] is not None
```

## TDD Implementation Checklist

### Week-by-Week Progression Checklist

#### Weeks 1-2: Foundation

- [ ] Set up pytest with healthcare-specific configuration
- [ ] Create test factories for health data models
- [ ] Establish Red-Green-Refactor discipline for utility functions
- [ ] Configure CI pipeline to run tests on every commit
- [ ] Team completes basic TDD training

#### Weeks 3-4: API Layer TDD

- [ ] All new API endpoints written test-first
- [ ] Authentication and authorization fully test-driven
- [ ] Input validation comprehensive and test-driven
- [ ] Error handling scenarios covered by tests
- [ ] API integration tests cover happy and error paths

#### Weeks 5-6: Business Logic TDD

- [ ] Health data processing logic is test-driven
- [ ] Clinical decision rules have comprehensive tests
- [ ] Business rule validation is test-driven
- [ ] Domain models have behavior-driven tests
- [ ] Service layer contracts are test-defined

#### Weeks 7-8: Integration TDD

- [ ] Database operations are contract-tested
- [ ] External service integrations use test doubles
- [ ] ML model integration is test-driven
- [ ] Message queue operations are tested
- [ ] File processing workflows are test-driven

#### Weeks 9-12: Advanced TDD

- [ ] Complete user stories built outside-in from acceptance tests
- [ ] Performance requirements verified through tests
- [ ] HIPAA compliance requirements tested automatically
- [ ] Security scenarios comprehensively tested
- [ ] Team demonstrates TDD mastery in code reviews

### Success Criteria by Risk Level

#### High-Risk Components (Security, Authentication, PII)

- [ ] 100% test coverage
- [ ] All edge cases covered
- [ ] Security scenarios tested
- [ ] Test-first development mandatory
- [ ] Code review requires TDD evidence

#### Medium-Risk Components (Business Logic, ML Integration)

- [ ] 90%+ test coverage
- [ ] Critical paths fully tested
- [ ] Error scenarios covered
- [ ] Test-first encouraged
- [ ] Regular TDD practice reviews

#### Low-Risk Components (Utilities, Formatting)

- [ ] 80%+ test coverage
- [ ] Happy path tested
- [ ] Basic error handling tested
- [ ] Test-after acceptable for simple functions
- [ ] Focus on complex logic

Remember: **TDD is not about testing everything - it's about designing software through tests that matter for healthcare safety, security, and reliability.**
