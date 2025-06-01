# Testing and Linting Integration Strategy

**Canonical approach for comprehensive quality assurance integration that catches issues before they become problems**

## Core Philosophy

**Quality Gates at Every Step**: Every code change must pass through automated quality checks before advancing. No compromises, no exceptions.

**Fail Fast, Fix Faster**: Catch issues at the earliest possible stage where they're cheapest to fix.

**Progressive Enhancement**: Start with basic checks and progressively add more sophisticated quality gates.

## Quality Assurance Stack

### Primary Tools Configuration

#### 1. MyPy (Type Checking)
```toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
show_error_codes = true

# Healthcare-specific type checking
[[tool.mypy.overrides]]
module = "clarity.models.health_data"
strict_optional = true
disallow_any_generics = true

[[tool.mypy.overrides]]
module = "clarity.ai.pat_model"
ignore_missing_imports = false
warn_return_any = true
```

#### 2. Ruff (Linting & Formatting)
```toml
# pyproject.toml
[tool.ruff]
target-version = "py311"
line-length = 88
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "S",   # bandit security
    "T20", # flake8-print
    "SIM", # flake8-simplify
    "RUF", # Ruff-specific rules
]
ignore = [
    "E501",  # line too long, handled by formatter
    "B008",  # do not perform function calls in argument defaults
    "S101",  # use of assert detected (we use pytest)
]

[tool.ruff.per-file-ignores]
"tests/**/*.py" = ["S101", "T20"]  # Allow assert and print in tests
"scripts/**/*.py" = ["T20"]       # Allow print in scripts

[tool.ruff.isort]
known-first-party = ["clarity"]
split-on-trailing-comma = true

[tool.ruff.bandit]
# Healthcare-specific security checks
assert-used = true
hardcoded-bind-all-interfaces = true
hardcoded-password-string = true
hardcoded-sql-expressions = true
```

#### 3. Bandit (Security Analysis)
```toml
# pyproject.toml
[tool.bandit]
exclude_dirs = ["tests", "build", "dist"]
skips = ["B101"]  # Skip assert_used for tests

# Healthcare-specific security rules
[tool.bandit.assert_used]
skips = ["**/tests/**", "**/test_*.py"]

[tool.bandit.hardcoded_password_string]
# No tolerance for hardcoded secrets in healthcare
word_list = ["password", "pass", "passwd", "pwd", "secret", "key", "token"]
```

#### 4. PyTest (Testing Framework)
```toml
# pyproject.toml
[tool.pytest.ini_options]
minversion = "7.0"
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=clarity",
    "--cov-report=term-missing:skip-covered",
    "--cov-report=html:htmlcov",
    "--cov-report=xml",
    "--cov-fail-under=90",
    "--durations=10",
    "-ra",
    "--tb=short"
]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "security: marks tests as security tests",
    "ml: marks tests as machine learning tests",
    "healthcare: marks tests as healthcare domain tests",
]
filterwarnings = [
    "error",
    "ignore::UserWarning",
    "ignore::DeprecationWarning",
]

# Healthcare-specific test configuration
asyncio_mode = "auto"
log_cli = true
log_cli_level = "INFO"
log_cli_format = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
```

#### 5. Coverage.py (Code Coverage)
```toml
# pyproject.toml
[tool.coverage.run]
source = ["clarity"]
omit = [
    "*/tests/*",
    "*/migrations/*",
    "*/venv/*",
    "*/env/*",
    "*/__pycache__/*",
    "*/node_modules/*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]

# Healthcare compliance requires high coverage
fail_under = 90
show_missing = true
skip_covered = false
```

## Integration Workflow

### Pre-Commit Hook Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: mypy
        name: mypy
        entry: mypy
        language: system
        types: [python]
        require_serial: true
        
      - id: ruff-check
        name: ruff-check
        entry: ruff check
        language: system
        types: [python]
        require_serial: true
        
      - id: ruff-format
        name: ruff-format
        entry: ruff format
        language: system
        types: [python]
        require_serial: true
        
      - id: bandit
        name: bandit
        entry: bandit
        language: system
        args: ['-r', 'clarity/', '-f', 'json']
        types: [python]
        
      - id: pytest-fast
        name: pytest-fast
        entry: pytest
        language: system
        args: ['-x', '-m', 'not slow']
        types: [python]
        pass_filenames: false
        always_run: true
```

### Development Quality Gates

#### Gate 1: Immediate Feedback (< 5 seconds)
```bash
#!/bin/bash
# scripts/quick-check.sh

echo "🔍 Running quick quality checks..."

# Format code automatically
echo "📝 Formatting code..."
ruff format .

# Basic linting
echo "🔧 Basic linting..."
ruff check --fix .

# Type checking on changed files only
echo "🏷️  Type checking changed files..."
git diff --name-only HEAD | grep '\.py$' | xargs mypy --follow-imports=silent

echo "✅ Quick checks complete!"
```

#### Gate 2: Pre-Commit Validation (< 30 seconds)
```bash
#!/bin/bash
# scripts/pre-commit-check.sh

echo "🛡️  Running pre-commit validation..."

# Full linting
echo "🔧 Full linting check..."
ruff check .

# Complete type checking
echo "🏷️  Complete type checking..."
mypy .

# Security scanning
echo "🔒 Security scanning..."
bandit -r clarity/ -f json -o bandit-report.json

# Fast test suite
echo "🧪 Running fast tests..."
pytest -x -m "not slow" --cov=clarity --cov-fail-under=80

echo "✅ Pre-commit validation complete!"
```

#### Gate 3: Full Quality Assurance (< 5 minutes)
```bash
#!/bin/bash
# scripts/full-qa.sh

echo "🔬 Running full quality assurance..."

# Complete test suite
echo "🧪 Running complete test suite..."
pytest --cov=clarity --cov-fail-under=90 --cov-report=html

# Integration tests
echo "🔗 Running integration tests..."
pytest tests/integration/ -v

# Security deep scan
echo "🔒 Deep security scanning..."
bandit -r clarity/ -f txt
safety check --json

# Performance benchmarks
echo "⚡ Performance benchmarks..."
pytest tests/performance/ -v --benchmark-only

echo "✅ Full QA complete!"
```

## Testing Strategy by Layer

### 1. API Layer Testing
```python
# tests/test_api/test_health_data.py
import pytest
from httpx import AsyncClient
from clarity.main import app
from clarity.models.health_data import HealthDataUpload

class TestHealthDataAPI:
    """Comprehensive API layer testing"""
    
    @pytest.mark.asyncio
    async def test_upload_valid_health_data(self, authenticated_client: AsyncClient):
        """Test successful health data upload"""
        health_data = {
            "data_type": "heart_rate",
            "values": [
                {"timestamp": "2024-01-01T12:00:00Z", "value": 72.0}
            ],
            "source": "apple_watch"
        }
        
        response = await authenticated_client.post(
            "/api/v1/health-data/upload",
            json=health_data
        )
        
        assert response.status_code == 202
        assert response.json()["status"] == "accepted"
        
    @pytest.mark.asyncio 
    async def test_upload_invalid_data_type(self, authenticated_client: AsyncClient):
        """Test rejection of invalid data types"""
        invalid_data = {
            "data_type": "invalid_type",
            "values": [],
            "source": "apple_watch"
        }
        
        response = await authenticated_client.post(
            "/api/v1/health-data/upload",
            json=invalid_data
        )
        
        assert response.status_code == 422
        assert "data_type" in response.json()["detail"][0]["loc"]
        
    @pytest.mark.security
    async def test_unauthorized_access_blocked(self, client: AsyncClient):
        """Ensure unauthorized requests are blocked"""
        response = await client.post(
            "/api/v1/health-data/upload",
            json={}
        )
        
        assert response.status_code == 401
        
    @pytest.mark.slow
    async def test_large_batch_upload_performance(self, authenticated_client: AsyncClient):
        """Test performance with large data batches"""
        large_batch = {
            "data_type": "heart_rate",
            "values": [
                {"timestamp": f"2024-01-01T{i:02d}:00:00Z", "value": 60.0 + i}
                for i in range(1000)
            ],
            "source": "apple_watch"
        }
        
        import time
        start_time = time.time()
        
        response = await authenticated_client.post(
            "/api/v1/health-data/upload",
            json=large_batch
        )
        
        duration = time.time() - start_time
        
        assert response.status_code == 202
        assert duration < 5.0  # Should process within 5 seconds
```

### 2. Business Logic Testing
```python
# tests/test_services/test_health_analyzer.py
import pytest
from clarity.services.health_analyzer import HealthAnalyzer
from clarity.models.health_data import HealthDataPoint

class TestHealthAnalyzer:
    """Business logic testing for health analysis"""
    
    @pytest.fixture
    def analyzer(self):
        return HealthAnalyzer()
        
    @pytest.fixture
    def sample_heart_rate_data(self):
        return [
            HealthDataPoint(timestamp="2024-01-01T12:00:00Z", value=72.0),
            HealthDataPoint(timestamp="2024-01-01T12:01:00Z", value=75.0),
            HealthDataPoint(timestamp="2024-01-01T12:02:00Z", value=78.0),
        ]
        
    def test_calculate_average_heart_rate(self, analyzer, sample_heart_rate_data):
        """Test average heart rate calculation"""
        average = analyzer.calculate_average_heart_rate(sample_heart_rate_data)
        assert average == 75.0
        
    def test_detect_heart_rate_anomalies(self, analyzer):
        """Test anomaly detection in heart rate data"""
        anomalous_data = [
            HealthDataPoint(timestamp="2024-01-01T12:00:00Z", value=72.0),
            HealthDataPoint(timestamp="2024-01-01T12:01:00Z", value=180.0),  # Anomaly
            HealthDataPoint(timestamp="2024-01-01T12:02:00Z", value=74.0),
        ]
        
        anomalies = analyzer.detect_anomalies(anomalous_data)
        assert len(anomalies) == 1
        assert anomalies[0].value == 180.0
        
    @pytest.mark.healthcare
    def test_phq9_score_calculation(self, analyzer):
        """Test PHQ-9 depression screening score calculation"""
        responses = [2, 1, 3, 2, 1, 0, 2, 1, 0]  # Sample PHQ-9 responses
        score = analyzer.calculate_phq9_score(responses)
        
        assert score == 12
        assert analyzer.classify_depression_severity(score) == "moderate"
        
    @pytest.mark.ml
    def test_actigraphy_preprocessing(self, analyzer):
        """Test actigraphy data preprocessing for ML model"""
        raw_actigraphy = [1, 0, 1, 1, 0, 0, 1] * 100  # Sample activity data
        
        processed = analyzer.preprocess_actigraphy(raw_actigraphy)
        
        assert len(processed) == len(raw_actigraphy)
        assert all(isinstance(x, float) for x in processed)
        assert -3 <= min(processed) <= max(processed) <= 3  # Z-score normalized
```

### 3. Data Layer Testing
```python
# tests/test_repositories/test_health_data_repository.py
import pytest
from clarity.repositories.health_data_repository import HealthDataRepository
from clarity.models.health_data import HealthDataUpload

class TestHealthDataRepository:
    """Data layer testing for health data persistence"""
    
    @pytest.fixture
    async def repository(self, test_firestore_client):
        return HealthDataRepository(test_firestore_client)
        
    @pytest.mark.asyncio
    async def test_store_health_data(self, repository):
        """Test storing health data in Firestore"""
        health_data = HealthDataUpload(
            user_id="test_user_123",
            data_type="heart_rate",
            values=[{"timestamp": "2024-01-01T12:00:00Z", "value": 72.0}],
            source="apple_watch"
        )
        
        document_id = await repository.store(health_data)
        
        assert document_id is not None
        assert len(document_id) > 0
        
    @pytest.mark.asyncio
    async def test_retrieve_user_health_data(self, repository):
        """Test retrieving health data for a specific user"""
        user_id = "test_user_123"
        
        # Store some test data first
        test_data = HealthDataUpload(
            user_id=user_id,
            data_type="steps",
            values=[{"timestamp": "2024-01-01T12:00:00Z", "value": 1000}],
            source="apple_watch"
        )
        await repository.store(test_data)
        
        # Retrieve the data
        retrieved_data = await repository.get_user_data(
            user_id=user_id,
            data_type="steps",
            start_date="2024-01-01",
            end_date="2024-01-02"
        )
        
        assert len(retrieved_data) >= 1
        assert retrieved_data[0].user_id == user_id
        
    @pytest.mark.security
    async def test_user_data_isolation(self, repository):
        """Ensure users can only access their own data"""
        # This test verifies proper user isolation in data access
        user1_data = await repository.get_user_data("user1", "heart_rate")
        user2_data = await repository.get_user_data("user2", "heart_rate")
        
        # Verify no cross-contamination
        user1_ids = {item.user_id for item in user1_data}
        user2_ids = {item.user_id for item in user2_data}
        
        assert "user2" not in user1_ids
        assert "user1" not in user2_ids
```

### 4. ML Model Testing
```python
# tests/test_ml/test_pat_model.py
import pytest
import torch
import numpy as np
from clarity.ml.pat_model import PATModel, preprocess_actigraphy

class TestPATModel:
    """Machine learning model testing for PAT (Pre-trained Actigraphy Transformer)"""
    
    @pytest.fixture
    def sample_actigraphy_data(self):
        """Generate sample actigraphy data for testing"""
        return np.random.randint(0, 2, size=(1440,))  # 24 hours of minute-level data
        
    @pytest.fixture
    def pat_model(self):
        """Load test PAT model"""
        return PATModel.load_test_model()
        
    @pytest.mark.ml
    def test_actigraphy_preprocessing(self, sample_actigraphy_data):
        """Test actigraphy data preprocessing"""
        processed = preprocess_actigraphy(sample_actigraphy_data)
        
        # Verify output shape and normalization
        assert processed.shape == sample_actigraphy_data.shape
        assert -3 <= processed.min() <= processed.max() <= 3  # Z-score bounds
        assert abs(processed.mean()) < 0.1  # Approximately zero mean
        
    @pytest.mark.ml
    def test_depression_prediction(self, pat_model, sample_actigraphy_data):
        """Test depression risk prediction from actigraphy"""
        processed_data = preprocess_actigraphy(sample_actigraphy_data)
        
        with torch.no_grad():
            prediction = pat_model.predict_depression_risk(processed_data)
            
        assert 0 <= prediction <= 1  # Valid probability
        assert isinstance(prediction, float)
        
    @pytest.mark.ml
    def test_model_reproducibility(self, pat_model, sample_actigraphy_data):
        """Test that model predictions are reproducible"""
        processed_data = preprocess_actigraphy(sample_actigraphy_data)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        prediction1 = pat_model.predict_depression_risk(processed_data)
        
        torch.manual_seed(42)
        prediction2 = pat_model.predict_depression_risk(processed_data)
        
        assert abs(prediction1 - prediction2) < 1e-6
        
    @pytest.mark.slow
    def test_batch_processing_performance(self, pat_model):
        """Test model performance with batch processing"""
        batch_size = 32
        batch_data = [
            preprocess_actigraphy(np.random.randint(0, 2, size=(1440,)))
            for _ in range(batch_size)
        ]
        
        import time
        start_time = time.time()
        
        predictions = pat_model.predict_batch(batch_data)
        
        duration = time.time() - start_time
        
        assert len(predictions) == batch_size
        assert duration < 10.0  # Should process batch within 10 seconds
        assert all(0 <= p <= 1 for p in predictions)
```

## Quality Metrics and Monitoring

### Coverage Requirements by Module

```python
# scripts/coverage_requirements.py
COVERAGE_REQUIREMENTS = {
    "clarity.api": 95,           # API endpoints must be thoroughly tested
    "clarity.services": 90,      # Business logic requires high coverage
    "clarity.repositories": 90,  # Data access critical for healthcare
    "clarity.ml": 85,           # ML models need comprehensive testing
    "clarity.auth": 100,        # Security components must be fully tested
    "clarity.models": 80,       # Data models need validation testing
    "clarity.utils": 75,        # Utility functions lower priority
}

def check_coverage_requirements():
    """Verify coverage meets module-specific requirements"""
    import coverage
    
    cov = coverage.Coverage()
    cov.load()
    
    for module, required_coverage in COVERAGE_REQUIREMENTS.items():
        actual_coverage = cov.report(show_missing=False, include=f"{module}/*")
        
        if actual_coverage < required_coverage:
            raise AssertionError(
                f"Coverage for {module} is {actual_coverage}%, "
                f"required {required_coverage}%"
            )
    
    print("✅ All coverage requirements met!")
```

### Automated Quality Dashboard

```python
# scripts/quality_dashboard.py
import json
import subprocess
from pathlib import Path

def generate_quality_report():
    """Generate comprehensive quality metrics report"""
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {}
    }
    
    # Test coverage
    coverage_result = subprocess.run(
        ["pytest", "--cov=clarity", "--cov-report=json"],
        capture_output=True, text=True
    )
    coverage_data = json.loads(Path("coverage.json").read_text())
    report["metrics"]["coverage"] = {
        "total": coverage_data["totals"]["percent_covered"],
        "by_module": {
            file: data["summary"]["percent_covered"]
            for file, data in coverage_data["files"].items()
        }
    }
    
    # Type checking
    mypy_result = subprocess.run(
        ["mypy", ".", "--json-report", "mypy-report"],
        capture_output=True, text=True
    )
    report["metrics"]["type_safety"] = {
        "errors": mypy_result.returncode == 0,
        "error_count": mypy_result.stderr.count("error:")
    }
    
    # Security scanning
    bandit_result = subprocess.run(
        ["bandit", "-r", "clarity/", "-f", "json"],
        capture_output=True, text=True
    )
    bandit_data = json.loads(bandit_result.stdout)
    report["metrics"]["security"] = {
        "high_severity": len([
            issue for issue in bandit_data["results"]
            if issue["issue_severity"] == "HIGH"
        ]),
        "medium_severity": len([
            issue for issue in bandit_data["results"]
            if issue["issue_severity"] == "MEDIUM"
        ])
    }
    
    # Code quality
    ruff_result = subprocess.run(
        ["ruff", "check", "--output-format=json", "."],
        capture_output=True, text=True
    )
    if ruff_result.stdout:
        ruff_data = json.loads(ruff_result.stdout)
        report["metrics"]["code_quality"] = {
            "violations": len(ruff_data),
            "by_category": {}
        }
        for violation in ruff_data:
            category = violation["code"][0]
            report["metrics"]["code_quality"]["by_category"][category] = (
                report["metrics"]["code_quality"]["by_category"].get(category, 0) + 1
            )
    else:
        report["metrics"]["code_quality"] = {"violations": 0}
    
    return report
```

## Continuous Integration Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/quality-assurance.yml
name: Quality Assurance

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  quality-check:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev,test]
    
    - name: Type checking with MyPy
      run: mypy .
    
    - name: Linting with Ruff
      run: ruff check .
    
    - name: Security scan with Bandit
      run: bandit -r clarity/ -f json -o bandit-report.json
    
    - name: Run tests with coverage
      run: |
        pytest --cov=clarity --cov-report=xml --cov-fail-under=90
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
    
    - name: Generate quality report
      run: python scripts/quality_dashboard.py
    
    - name: Archive quality artifacts
      uses: actions/upload-artifact@v3
      with:
        name: quality-reports
        path: |
          htmlcov/
          bandit-report.json
          mypy-report/
```

### Local Development Script

```bash
#!/bin/bash
# scripts/dev-check.sh

set -e

echo "🚀 Starting development quality check..."

# Install development dependencies if needed
if ! command -v ruff &> /dev/null; then
    echo "📦 Installing development dependencies..."
    pip install -e .[dev,test]
fi

# Create quality report directory
mkdir -p reports

# 1. Format code
echo "📝 Formatting code..."
ruff format .

# 2. Fix linting issues automatically
echo "🔧 Auto-fixing linting issues..."
ruff check --fix .

# 3. Type checking
echo "🏷️  Type checking..."
mypy . --html-report reports/mypy || {
    echo "❌ Type checking failed. Check reports/mypy/index.html"
    exit 1
}

# 4. Security scanning
echo "🔒 Security scanning..."
bandit -r clarity/ -f json -o reports/bandit-report.json || {
    echo "⚠️  Security issues found. Check reports/bandit-report.json"
}

# 5. Run tests with coverage
echo "🧪 Running tests..."
pytest --cov=clarity --cov-report=html:reports/coverage --cov-fail-under=90 || {
    echo "❌ Tests failed or coverage below 90%. Check reports/coverage/index.html"
    exit 1
}

# 6. Generate quality dashboard
echo "📊 Generating quality dashboard..."
python scripts/quality_dashboard.py > reports/quality-report.json

echo "✅ All quality checks passed!"
echo "📈 View reports at: file://$(pwd)/reports/"
```

## Quality Gates Implementation

### Pre-Push Quality Gate

```python
#!/usr/bin/env python3
# scripts/pre_push_gate.py
"""
Pre-push quality gate that prevents pushing code that doesn't meet standards.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: list[str]) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr"""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def check_type_safety() -> bool:
    """Verify type safety with MyPy"""
    print("🏷️  Checking type safety...")
    exit_code, stdout, stderr = run_command(["mypy", "."])
    
    if exit_code != 0:
        print(f"❌ Type checking failed:\n{stderr}")
        return False
    
    print("✅ Type checking passed")
    return True

def check_security() -> bool:
    """Verify security with Bandit"""
    print("🔒 Checking security...")
    exit_code, stdout, stderr = run_command([
        "bandit", "-r", "clarity/", "-f", "json"
    ])
    
    if exit_code != 0:
        import json
        try:
            report = json.loads(stdout)
            high_severity = [
                issue for issue in report["results"]
                if issue["issue_severity"] == "HIGH"
            ]
            if high_severity:
                print(f"❌ High severity security issues found: {len(high_severity)}")
                return False
        except json.JSONDecodeError:
            print(f"❌ Security scan failed:\n{stderr}")
            return False
    
    print("✅ Security check passed")
    return True

def check_test_coverage() -> bool:
    """Verify test coverage meets requirements"""
    print("🧪 Checking test coverage...")
    exit_code, stdout, stderr = run_command([
        "pytest", "--cov=clarity", "--cov-fail-under=90", "-q"
    ])
    
    if exit_code != 0:
        print(f"❌ Test coverage below 90% or tests failed:\n{stderr}")
        return False
    
    print("✅ Test coverage check passed")
    return True

def main():
    """Run all quality gates"""
    print("🛡️  Running pre-push quality gates...")
    
    checks = [
        check_type_safety,
        check_security,
        check_test_coverage,
    ]
    
    for check in checks:
        if not check():
            print("❌ Quality gate failed. Push blocked.")
            sys.exit(1)
    
    print("✅ All quality gates passed. Push allowed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
```

### IDE Integration (VS Code)

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.mypyEnabled": true,
  "python.linting.banditEnabled": true,
  "python.formatting.provider": "none",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "--cov=clarity",
    "--cov-report=html",
    "-v"
  ],
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true,
      "source.fixAll": true
    }
  },
  "ruff.args": ["--config", "pyproject.toml"],
  "mypy-type-checker.args": ["--config-file", "pyproject.toml"],
  "files.associations": {
    "*.toml": "toml"
  },
  "testing.python.pytest.addArgs": [
    "--cov=clarity",
    "--cov-report=html:htmlcov"
  ]
}
```

## Performance and Load Testing

### Performance Test Suite

```python
# tests/performance/test_api_performance.py
import pytest
import asyncio
import time
from httpx import AsyncClient

class TestAPIPerformance:
    """Performance testing for API endpoints"""
    
    @pytest.mark.slow
    async def test_health_data_upload_throughput(self, authenticated_client: AsyncClient):
        """Test API throughput under load"""
        sample_data = {
            "data_type": "heart_rate",
            "values": [{"timestamp": "2024-01-01T12:00:00Z", "value": 72.0}],
            "source": "apple_watch"
        }
        
        # Concurrent requests
        concurrent_requests = 50
        start_time = time.time()
        
        tasks = [
            authenticated_client.post("/api/v1/health-data/upload", json=sample_data)
            for _ in range(concurrent_requests)
        ]
        
        responses = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # Verify all requests succeeded
        successful_requests = sum(1 for r in responses if r.status_code == 202)
        
        assert successful_requests == concurrent_requests
        assert duration < 10.0  # Should handle 50 requests in under 10 seconds
        
        # Calculate throughput
        throughput = concurrent_requests / duration
        assert throughput > 5.0  # Should handle at least 5 requests per second
        
    @pytest.mark.benchmark
