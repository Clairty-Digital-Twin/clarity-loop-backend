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
