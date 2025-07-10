# CLARITY Backend Test Harness Guide

## Overview

This comprehensive test harness provides a robust foundation for testing the CLARITY backend, ensuring safe refactoring and maintaining system reliability.

## Architecture

```
tests/
├── fixtures/               # Reusable test fixtures
│   ├── aws_fixtures.py    # AWS service mocks (DynamoDB, S3, Cognito)
│   └── data_factories.py  # Test data generation utilities
├── performance/           # Performance testing
│   └── locustfile.py     # Load testing scenarios
├── unit/                 # Unit tests
├── integration/          # Integration tests
├── conftest.py          # Global pytest configuration
└── pytest.ini           # Pytest settings
```

## Key Components

### 1. AWS Service Fixtures (`fixtures/aws_fixtures.py`)

Provides isolated AWS service mocks using `moto`:

- **DynamoDB Fixtures**:
  - `dynamodb_mock`: Mocked DynamoDB resource
  - `dynamodb_table`: Pre-configured test table with GSI
  - `dynamodb_batch_writer`: Helper for batch operations

- **S3 Fixtures**:
  - `s3_mock`: Mocked S3 client
  - `s3_bucket`: Test bucket with lifecycle policies
  - `s3_batch_uploader`: Batch upload helper

- **Cognito Fixtures**:
  - `cognito_mock`: Mocked Cognito IDP client
  - `cognito_user_pool`: Test user pool with client
  - `cognito_test_user`: Pre-created test user

### 2. Test Data Factories (`fixtures/data_factories.py`)

Factory-based test data generation:

- `UserFactory`: Generate user profiles
- `HealthMetricFactory`: Create health metrics
- `ActivityDataFactory`: Generate activity data
- `SleepDataFactory`: Create sleep patterns
- `PATPredictionFactory`: ML prediction data
- `AnalysisReportFactory`: Analysis reports

Batch generators for realistic datasets:
- `generate_user_batch()`
- `generate_health_metrics_batch()`
- `generate_activity_timeline()`
- `generate_test_dataset()`

### 3. Performance Testing (`performance/locustfile.py`)

Load testing scenarios:

- `ClarityAPIUser`: Standard user behavior
- `ClarityWebSocketUser`: Real-time connections
- `ClarityHeavyUser`: Power user patterns

### 4. Test Configuration (`pytest.ini`)

- Coverage target: 65% (increasing to 85%)
- Test markers for categorization
- Parallel execution support
- Comprehensive reporting

## Usage

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
pytest --cov=src/clarity --cov-report=html

# Run specific categories
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests
pytest -m "not slow"     # Skip slow tests

# Parallel execution
pytest -n auto          # Auto-detect CPU cores
pytest -n 4            # Use 4 workers
```

### Using Fixtures

```python
# Example: Testing with DynamoDB
def test_user_creation(dynamodb_table, dynamodb_batch_writer):
    # Use batch writer to seed data
    users = generate_user_batch(5)
    dynamodb_batch_writer(users)
    
    # Test your service
    response = dynamodb_table.scan()
    assert response['Count'] == 5

# Example: Testing with multiple AWS services
def test_full_workflow(dynamodb_table, s3_bucket, cognito_test_user):
    # User is already created in Cognito
    username = cognito_test_user['username']
    
    # Store user data in DynamoDB
    dynamodb_table.put_item(Item={'pk': f'USER#{username}'})
    
    # Upload file to S3
    s3_bucket['client'].put_object(
        Bucket=s3_bucket['name'],
        Key=f'users/{username}/data.json',
        Body=json.dumps({'test': 'data'})
    )
```

### Performance Testing

```bash
# Run load test (default: 10 users)
locust -f tests/performance/locustfile.py --host=http://localhost:8000

# Headless mode with specific parameters
locust -f tests/performance/locustfile.py \
    --host=http://localhost:8000 \
    --headless \
    --users 100 \
    --spawn-rate 10 \
    --run-time 5m

# Generate HTML report
locust -f tests/performance/locustfile.py \
    --host=http://localhost:8000 \
    --html performance_report.html
```

## Test Categories

### Markers

- `@pytest.mark.unit`: Isolated unit tests
- `@pytest.mark.integration`: Service integration tests
- `@pytest.mark.functional`: End-to-end tests
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.slow`: Tests taking >5 seconds
- `@pytest.mark.smoke`: Critical smoke tests
- `@pytest.mark.aws`: Tests requiring AWS services
- `@pytest.mark.ml`: Machine learning tests
- `@pytest.mark.critical`: Must-pass tests

### Example Test Structure

```python
@pytest.mark.unit
@pytest.mark.aws
class TestDynamoDBService:
    """Test DynamoDB service operations."""
    
    @pytest.fixture(autouse=True)
    def setup(self, dynamodb_table):
        """Set up test environment."""
        self.table = dynamodb_table
        self.service = DynamoDBService(table=self.table)
    
    def test_create_item(self):
        """Test item creation."""
        item = {"pk": "TEST#123", "sk": "METADATA"}
        self.service.create_item(item)
        
        response = self.table.get_item(Key={"pk": "TEST#123", "sk": "METADATA"})
        assert "Item" in response
```

## Baseline Metrics

Current baseline metrics to maintain:

- **Test Coverage**: 65% (target: 85%)
- **Test Execution Time**: <5 minutes for unit tests
- **API Response Time**: <200ms for 95th percentile
- **Memory Usage**: <500MB during test runs
- **Test Reliability**: >99% pass rate

## CI/CD Integration

The test harness is integrated with CI/CD:

```yaml
# Example GitHub Actions configuration
test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -e .[test]
    - name: Run tests
      run: |
        pytest --cov=src/clarity --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Best Practices

1. **Isolation**: Always use mocked AWS services in tests
2. **Determinism**: Use fixed seeds for random data
3. **Cleanup**: Fixtures handle cleanup automatically
4. **Parallelism**: Tests should be independent for parallel execution
5. **Documentation**: Document complex test scenarios
6. **Performance**: Monitor test execution time

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure test dependencies are installed: `pip install -e .[test]`
2. **AWS Credential Errors**: Fixtures set test credentials automatically
3. **Flaky Tests**: Check for shared state or timing issues
4. **Slow Tests**: Use parallel execution or mark as `@pytest.mark.slow`

### Debug Mode

```bash
# Run with verbose output
pytest -vv

# Show print statements
pytest -s

# Debug specific test
pytest -k test_name --pdb
```

## Next Steps

1. Run baseline metrics: `pytest --cov --benchmark-only`
2. Identify gaps in test coverage: `coverage html && open htmlcov/index.html`
3. Add missing unit tests for uncovered code
4. Set up continuous monitoring of test metrics