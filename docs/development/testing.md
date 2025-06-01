# Testing Strategy

This document outlines the comprehensive testing strategy for the Clarity Loop Backend, ensuring high-quality, reliable, and secure health data processing.

## Testing Philosophy

### Core Testing Principles
- **Test-Driven Development (TDD)**: Write tests before implementation
- **Async-First Testing**: All tests designed for async operations
- **Health Data Security**: HIPAA-compliant test data handling
- **AI Model Validation**: Comprehensive ML pipeline testing
- **Integration Focused**: End-to-end testing of core user journeys

### Testing Pyramid
```
                    ┌─────────────────┐
                    │   E2E Tests     │ ← 10% (Critical user flows)
                    │   (Playwright)  │
                ┌───┴─────────────────┴───┐
                │   Integration Tests     │ ← 20% (API + Database + ML)
                │   (pytest + Firebase)  │
            ┌───┴─────────────────────────┴───┐
            │        Unit Tests               │ ← 70% (Business logic)
            │   (pytest + mocks + fixtures)  │
            └─────────────────────────────────┘
```

## Test Categories

### 1. Unit Tests (70% of test suite)
**Purpose**: Test individual functions and classes in isolation

**Coverage Areas**:
- Pydantic model validation
- Business logic functions
- Data transformation utilities
- Authentication helpers
- ML model preprocessing

**Example Structure**:
```python
# tests/unit/test_health_data_models.py
import pytest
from clarity.models.health_data import HealthDataPoint, DataQualityMetrics

class TestHealthDataPoint:
    def test_valid_heart_rate_data(self):
        """Test valid heart rate data creation."""
        data = HealthDataPoint(
            user_id="user_123",
            data_type="heart_rate",
            value=72.5,
            timestamp="2024-01-01T12:00:00Z",
            source="apple_watch"
        )
        assert data.value == 72.5
        assert data.data_type == "heart_rate"
    
    def test_invalid_heart_rate_range(self):
        """Test validation of heart rate ranges."""
        with pytest.raises(ValueError, match="Heart rate must be between"):
            HealthDataPoint(
                user_id="user_123",
                data_type="heart_rate",
                value=300,  # Invalid: too high
                timestamp="2024-01-01T12:00:00Z",
                source="apple_watch"
            )
    
    @pytest.mark.asyncio
    async def test_data_quality_assessment(self):
        """Test data quality scoring algorithm."""
        metrics = DataQualityMetrics()
        score = await metrics.assess_quality([
            {"value": 72, "timestamp": "2024-01-01T12:00:00Z"},
            {"value": 74, "timestamp": "2024-01-01T12:01:00Z"},
        ])
        assert 0.8 <= score <= 1.0
```

### 2. Integration Tests (20% of test suite)
**Purpose**: Test component interactions and external service integration

**Coverage Areas**:
- API endpoints with authentication
- Database operations with Firestore
- ML model inference pipeline
- Pub/Sub message processing
- Firebase Authentication flow

**Example Structure**:
```python
# tests/integration/test_health_data_api.py
import pytest
from httpx import AsyncClient
from clarity.main import app

class TestHealthDataAPI:
    @pytest.mark.asyncio
    @pytest.mark.requires_firebase
    async def test_upload_health_data_authenticated(
        self, authenticated_client: AsyncClient, sample_health_data
    ):
        """Test health data upload with valid authentication."""
        response = await authenticated_client.post(
            "/api/v1/health-data/upload",
            json=sample_health_data
        )
        assert response.status_code == 202
        assert "processing_id" in response.json()
    
    @pytest.mark.asyncio
    async def test_upload_health_data_unauthenticated(
        self, client: AsyncClient, sample_health_data
    ):
        """Test health data upload without authentication."""
        response = await client.post(
            "/api/v1/health-data/upload",
            json=sample_health_data
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Authentication required"
    
    @pytest.mark.asyncio
    @pytest.mark.requires_gcp
    async def test_data_processing_pipeline(
        self, authenticated_client: AsyncClient, firestore_emulator
    ):
        """Test end-to-end data processing pipeline."""
        # Upload data
        upload_response = await authenticated_client.post(
            "/api/v1/health-data/upload",
            json={"data_type": "heart_rate", "values": [72, 74, 76]}
        )
        processing_id = upload_response.json()["processing_id"]
        
        # Wait for processing (or mock async processing)
        await asyncio.sleep(1)
        
        # Verify data was processed and stored
        insights_response = await authenticated_client.get(
            f"/api/v1/insights/daily?processing_id={processing_id}"
        )
        assert insights_response.status_code == 200
        assert "heart_rate_analysis" in insights_response.json()
```

### 3. End-to-End Tests (10% of test suite)
**Purpose**: Test complete user journeys across the entire system

**Coverage Areas**:
- User registration and authentication flow
- Health data upload → processing → insights generation
- Real-time chat with health AI
- Data export and deletion workflows

**Example Structure**:
```python
# tests/e2e/test_user_journey.py
import pytest
from playwright.async_api import async_playwright

class TestCompleteUserJourney:
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_new_user_onboarding_to_insights(self):
        """Test complete user journey from registration to insights."""
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            # 1. User registration (mocked iOS app flow)
            registration_data = {
                "email": "test@example.com",
                "firebase_token": "mock_firebase_token"
            }
            
            # 2. Health data sync simulation
            health_data = generate_sample_week_data()
            
            # 3. Verify insights generation
            insights_response = await simulate_insights_request(
                page, health_data
            )
            
            assert "weekly_summary" in insights_response
            assert "recommendations" in insights_response
            
            await browser.close()
```

## Test Configuration

### Pytest Configuration
```toml
# pyproject.toml [tool.pytest.ini_options] - already included in main pyproject.toml
```

### Environment Setup
```python
# tests/conftest.py
import pytest
import asyncio
from httpx import AsyncClient
from clarity.main import app
from clarity.config import get_settings
from tests.utils.firebase_emulator import FirebaseEmulator
from tests.utils.fixtures import *

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def client():
    """Create an unauthenticated test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
async def authenticated_client(mock_firebase_auth):
    """Create an authenticated test client."""
    app.dependency_overrides[get_current_user] = mock_firebase_auth
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()

@pytest.fixture(scope="session")
async def firestore_emulator():
    """Start Firestore emulator for integration tests."""
    emulator = FirebaseEmulator()
    await emulator.start()
    yield emulator
    await emulator.stop()

@pytest.fixture
def sample_health_data():
    """Generate sample health data for testing."""
    return {
        "data_type": "heart_rate",
        "values": [
            {"value": 72, "timestamp": "2024-01-01T12:00:00Z"},
            {"value": 74, "timestamp": "2024-01-01T12:01:00Z"},
            {"value": 76, "timestamp": "2024-01-01T12:02:00Z"},
        ],
        "source": "apple_watch",
        "source_version": "10.0"
    }
```

## Test Data Management

### HIPAA-Compliant Test Data
```python
# tests/utils/test_data_generator.py
import faker
from datetime import datetime, timedelta
from typing import List, Dict

class HealthDataGenerator:
    """Generate realistic but fake health data for testing."""
    
    def __init__(self):
        self.fake = faker.Faker()
        faker.Faker.seed(42)  # Deterministic test data
    
    def generate_user_profile(self) -> Dict:
        """Generate a fake user profile."""
        return {
            "user_id": f"test_user_{self.fake.uuid4()}",
            "email": self.fake.email(),
            "date_of_birth": self.fake.date_of_birth(minimum_age=18, maximum_age=80),
            "gender": self.fake.random_element(["male", "female", "other"]),
            "height_cm": self.fake.random_int(150, 200),
            "weight_kg": self.fake.random_int(50, 120),
            "activity_level": self.fake.random_element(["low", "moderate", "high"])
        }
    
    def generate_heart_rate_series(
        self, 
        days: int = 7, 
        user_profile: Dict = None
    ) -> List[Dict]:
        """Generate realistic heart rate data series."""
        data = []
        base_hr = 70 if not user_profile else self._calculate_base_hr(user_profile)
        
        for day in range(days):
            date = datetime.now() - timedelta(days=days-day)
            
            # Generate 24 hours of heart rate data
            for hour in range(24):
                # Simulate circadian rhythm
                time_factor = self._circadian_factor(hour)
                hr_value = base_hr * time_factor + self.fake.random_int(-5, 5)
                
                data.append({
                    "value": max(50, min(180, hr_value)),  # Realistic bounds
                    "timestamp": date.replace(hour=hour).isoformat(),
                    "data_type": "heart_rate",
                    "source": "test_device"
                })
        
        return data
    
    def _circadian_factor(self, hour: int) -> float:
        """Calculate heart rate factor based on time of day."""
        if 6 <= hour <= 22:  # Awake hours
            return 1.0 + 0.1 * np.sin((hour - 6) * np.pi / 16)
        else:  # Sleep hours
            return 0.7
    
    def _calculate_base_hr(self, profile: Dict) -> int:
        """Calculate baseline heart rate based on user profile."""
        base = 70
        
        # Age factor
        age = (datetime.now().date() - profile["date_of_birth"]).days // 365
        base += (age - 30) * 0.2
        
        # Activity level factor
        activity_factors = {"low": 5, "moderate": 0, "high": -10}
        base += activity_factors.get(profile["activity_level"], 0)
        
        return int(max(50, min(100, base)))
```

### Test Database Management
```python
# tests/utils/database.py
import asyncio
from google.cloud import firestore
from clarity.config import get_settings

class TestDatabase:
    """Manage test database state and cleanup."""
    
    def __init__(self):
        self.db = firestore.AsyncClient(project="test-project")
        self.created_documents = []
    
    async def create_test_user(self, user_data: Dict) -> str:
        """Create a test user and track for cleanup."""
        doc_ref = self.db.collection("users").document()
        await doc_ref.set(user_data)
        self.created_documents.append(doc_ref)
        return doc_ref.id
    
    async def create_test_health_data(
        self, 
        user_id: str, 
        health_data: List[Dict]
    ) -> List[str]:
        """Create test health data and track for cleanup."""
        doc_ids = []
        
        for data_point in health_data:
            doc_ref = self.db.collection("health_data").document()
            data_point["user_id"] = user_id
            await doc_ref.set(data_point)
            self.created_documents.append(doc_ref)
            doc_ids.append(doc_ref.id)
        
        return doc_ids
    
    async def cleanup(self):
        """Clean up all test data."""
        cleanup_tasks = [
            doc_ref.delete() for doc_ref in self.created_documents
        ]
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        self.created_documents.clear()
```

## ML Model Testing

### Actigraphy Transformer Testing
```python
# tests/ml/test_actigraphy_transformer.py
import pytest
import torch
import numpy as np
from clarity.ml.actigraphy_transformer import ActigraphyTransformer
from clarity.ml.preprocessing import HealthDataPreprocessor

class TestActigraphyTransformer:
    @pytest.fixture
    def model(self):
        """Load test model or create mock."""
        return ActigraphyTransformer.load_test_model()
    
    @pytest.fixture
    def sample_actigraphy_data(self):
        """Generate sample actigraphy data."""
        # 7 days of minute-by-minute data
        return np.random.rand(7 * 24 * 60, 3)  # 3 axes
    
    @pytest.mark.ml
    def test_model_input_validation(self, model, sample_actigraphy_data):
        """Test model input validation and preprocessing."""
        preprocessor = HealthDataPreprocessor()
        
        # Test valid input
        processed_data = preprocessor.prepare_actigraphy_input(
            sample_actigraphy_data
        )
        assert processed_data.shape[0] == 7  # 7 days
        assert processed_data.shape[1] == 1440  # minutes per day
        assert processed_data.shape[2] == 3  # 3 axes
    
    @pytest.mark.ml
    def test_model_inference(self, model, sample_actigraphy_data):
        """Test model inference pipeline."""
        preprocessor = HealthDataPreprocessor()
        processed_data = preprocessor.prepare_actigraphy_input(
            sample_actigraphy_data
        )
        
        with torch.no_grad():
            output = model(torch.tensor(processed_data, dtype=torch.float32))
        
        # Verify output structure
        assert "sleep_stages" in output
        assert "activity_patterns" in output
        assert "circadian_rhythm" in output
        
        # Verify output ranges
        sleep_stages = output["sleep_stages"]
        assert torch.all(sleep_stages >= 0) and torch.all(sleep_stages <= 4)
    
    @pytest.mark.ml
    @pytest.mark.slow
    def test_model_performance_benchmark(self, model):
        """Test model performance against benchmarks."""
        benchmark_data = load_benchmark_dataset()
        
        total_inference_time = 0
        correct_predictions = 0
        
        for sample, expected in benchmark_data:
            start_time = time.time()
            prediction = model.predict(sample)
            inference_time = time.time() - start_time
            
            total_inference_time += inference_time
            if self._predictions_match(prediction, expected):
                correct_predictions += 1
        
        # Performance assertions
        avg_inference_time = total_inference_time / len(benchmark_data)
        accuracy = correct_predictions / len(benchmark_data)
        
        assert avg_inference_time < 2.0  # Max 2 seconds per inference
        assert accuracy > 0.85  # Min 85% accuracy
```

### Gemini AI Integration Testing
```python
# tests/ml/test_gemini_integration.py
import pytest
from unittest.mock import AsyncMock, patch
from clarity.ml.gemini_client import GeminiInsightsGenerator

class TestGeminiIntegration:
    @pytest.fixture
    def gemini_client(self):
        """Create Gemini client with test configuration."""
        return GeminiInsightsGenerator(
            api_key="test_key",
            model_name="gemini-2.0-flash-exp"
        )
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_health_insights_generation(
        self, gemini_client, sample_health_analysis
    ):
        """Test health insights generation from analysis data."""
        with patch('google.generativeai.generate_text') as mock_generate:
            mock_generate.return_value = AsyncMock(
                text="Based on your heart rate data, you show excellent cardiovascular health..."
            )
            
            insights = await gemini_client.generate_insights(
                user_profile={"age": 30, "activity_level": "moderate"},
                health_analysis=sample_health_analysis,
                insight_type="daily_summary"
            )
            
            assert "cardiovascular" in insights.text.lower()
            assert insights.confidence > 0.7
            assert insights.recommendations is not None
    
    @pytest.mark.asyncio
    async def test_insights_rate_limiting(self, gemini_client):
        """Test rate limiting and retry logic."""
        with patch('google.generativeai.generate_text') as mock_generate:
            # Simulate rate limit error
            mock_generate.side_effect = [
                Exception("Rate limit exceeded"),
                AsyncMock(text="Retry successful")
            ]
            
            insights = await gemini_client.generate_insights(
                user_profile={},
                health_analysis={},
                insight_type="daily_summary"
            )
            
            assert insights.text == "Retry successful"
            assert mock_generate.call_count == 2
```

## Performance Testing

### Load Testing Configuration
```python
# tests/performance/test_api_load.py
import pytest
import asyncio
from httpx import AsyncClient
import time

class TestAPIPerformance:
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_health_data_upload_concurrent(self):
        """Test concurrent health data uploads."""
        async def upload_data(client, data):
            response = await client.post("/api/v1/health-data/upload", json=data)
            return response.status_code, response.elapsed.total_seconds()
        
        # Simulate 100 concurrent uploads
        clients = [AsyncClient(app=app, base_url="http://test") for _ in range(100)]
        test_data = generate_sample_health_data()
        
        start_time = time.time()
        tasks = [upload_data(client, test_data) for client in clients]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Performance assertions
        success_count = sum(1 for status, _ in results if status == 202)
        avg_response_time = sum(elapsed for _, elapsed in results) / len(results)
        
        assert success_count >= 95  # 95% success rate
        assert avg_response_time < 1.0  # Average response time under 1 second
        assert total_time < 30.0  # Total time under 30 seconds
        
        # Cleanup
        for client in clients:
            await client.aclose()
```

## Test Automation

### CI/CD Integration
```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      firestore-emulator:
        image: google/cloud-sdk:latest
        ports:
          - 8080:8080
        options: --entrypoint gcloud beta emulators firestore start --host-port=0.0.0.0:8080
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
        
      - name: Install dependencies
        run: uv sync --extra dev
        
      - name: Run unit tests
        run: uv run pytest tests/unit -v --cov=src/clarity --cov-report=xml
        
      - name: Run integration tests
        run: uv run pytest tests/integration -v -m "not slow"
        env:
          FIRESTORE_EMULATOR_HOST: localhost:8080
          
      - name: Run ML tests
        run: uv run pytest tests/ml -v -m "not slow"
        
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
        
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        
  - repo: local
    hooks:
      - id: pytest-unit
        name: pytest-unit
        entry: uv run pytest tests/unit
        language: system
        pass_filenames: false
```

## Testing Utilities

### Custom Test Decorators
```python
# tests/utils/decorators.py
import functools
import pytest
from typing import Callable

def requires_firebase_emulator(func: Callable) -> Callable:
    """Skip test if Firebase emulator is not available."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        if not is_firebase_emulator_running():
            pytest.skip("Firebase emulator not available")
        return await func(*args, **kwargs)
    return wrapper

def mock_ml_model(model_name: str):
    """Decorator to mock ML model responses."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            with patch(f'clarity.ml.{model_name}') as mock_model:
                mock_model.predict.return_value = generate_mock_prediction()
                return await func(*args, **kwargs)
        return wrapper
    return decorator

def with_test_user(user_type: str = "standard"):
    """Decorator to create and cleanup test user."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            test_db = TestDatabase()
            user_data = generate_test_user(user_type)
            user_id = await test_db.create_test_user(user_data)
            
            try:
                return await func(*args, user_id=user_id, **kwargs)
            finally:
                await test_db.cleanup()
        return wrapper
    return decorator
```

## Test Reporting

### Coverage Requirements
- **Minimum Coverage**: 80% overall
- **Critical Paths**: 95% coverage for authentication, health data processing, AI insights
- **ML Models**: 90% coverage for preprocessing and postprocessing
- **API Endpoints**: 100% coverage for all public endpoints

### Test Metrics Dashboard
```python
# scripts/test_metrics.py
import json
from pathlib import Path

def generate_test_report():
    """Generate comprehensive test metrics report."""
    coverage_data = load_coverage_data()
    performance_data = load_performance_data()
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "coverage": {
            "overall": coverage_data["overall"],
            "by_module": coverage_data["modules"],
            "critical_paths": coverage_data["critical"]
        },
        "performance": {
            "api_response_times": performance_data["api"],
            "ml_inference_times": performance_data["ml"],
            "concurrent_user_capacity": performance_data["load"]
        },
        "test_counts": {
            "unit": count_tests("tests/unit"),
            "integration": count_tests("tests/integration"),
            "e2e": count_tests("tests/e2e"),
            "ml": count_tests("tests/ml")
        }
    }
    
    # Save report
    Path("reports/test_metrics.json").write_text(json.dumps(report, indent=2))
    
    return report
```

This comprehensive testing strategy ensures the Clarity Loop Backend maintains high quality, security, and performance standards while supporting the core functionality of health data chat with AI insights.
