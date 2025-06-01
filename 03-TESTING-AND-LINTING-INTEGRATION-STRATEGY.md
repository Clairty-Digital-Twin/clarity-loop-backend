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
