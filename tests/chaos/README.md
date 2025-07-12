# Chaos Testing Module

This module contains chaos tests for the CLARITY platform, focusing on system resilience and failure handling.

## Overview

The chaos tests simulate various failure scenarios to ensure the system degrades gracefully and recovers properly. All tests are designed to be **non-destructive** and safe for CI/CD pipelines.

## Test Categories

### Model Corruption Tests (`test_model_corruption.py`)

Tests various ML model corruption scenarios:

1. **Corrupted Model Files**: Tests handling when model files are corrupted or unreadable
2. **Checksum Mismatches**: Verifies detection of tampered or corrupted models via checksum validation
3. **Graceful Degradation**: Ensures service returns appropriate 503 responses when models are unavailable
4. **Circuit Breaker**: Tests circuit breaker activation after repeated failures
5. **Performance Degradation**: Monitors system performance under corruption conditions
6. **Concurrent Corruption**: Tests handling of corruption under concurrent load

## Running Chaos Tests

### Run all chaos tests:
```bash
pytest tests/chaos/
```

### Run only fast chaos tests (skip slow tests):
```bash
pytest tests/chaos/ -m "not slow"
```

### Run with detailed output:
```bash
pytest tests/chaos/ -v --tb=long
```

### Run specific test:
```bash
pytest tests/chaos/test_model_corruption.py::TestModelCorruption::test_circuit_breaker_activation
```

## CI/CD Integration

The chaos tests are designed to run safely in CI/CD pipelines:

1. **Non-destructive**: Tests use temporary directories and mocked services
2. **Fast execution**: Most tests complete in < 1 second
3. **Isolated**: Each test cleans up after itself
4. **Deterministic**: Tests produce consistent results

### GitHub Actions Example:

```yaml
- name: Run Chaos Tests
  run: |
    pytest tests/chaos/ -m "not slow" --junit-xml=chaos-test-results.xml
  continue-on-error: false
```

## Key Features Tested

### 1. Model Integrity
- SHA-256 checksum verification
- File corruption detection
- Version mismatch handling

### 2. Circuit Breaker
- Failure threshold: 5 attempts
- Recovery timeout: 60 seconds
- Graceful service degradation

### 3. Performance Monitoring
- Load time tracking
- Concurrent request handling
- Resource usage under failure

### 4. Error Responses
- Proper 503 Service Unavailable responses
- Detailed error logging
- Metrics collection

## Safety Guarantees

All chaos tests follow these principles:

1. **No production data access**: Tests use mock data only
2. **No external service calls**: All external dependencies are mocked
3. **Temporary file usage**: Tests create and clean up temp directories
4. **Resource cleanup**: All resources are properly released
5. **Timeout protection**: Tests have 30-second timeout limits

## Metrics and Monitoring

The tests verify that the following metrics are properly collected:

- `prediction_success_total`: Successful model predictions
- `prediction_failure_total`: Failed model predictions
- Circuit breaker state transitions
- Model load times
- Checksum verification times

## Adding New Chaos Tests

When adding new chaos tests:

1. Use the `@pytest.mark.chaos` decorator
2. Ensure tests are non-destructive
3. Mock all external dependencies
4. Clean up all resources in fixtures
5. Add appropriate timeout limits
6. Document the failure scenario being tested

Example template:

```python
@pytest.mark.chaos
@pytest.mark.non_destructive
async def test_new_failure_scenario(self, temp_dir):
    """Test description of failure scenario."""
    # Setup
    # Inject failure
    # Verify graceful handling
    # Cleanup (automatic with fixtures)
```

## Troubleshooting

### Tests timing out
- Check for infinite loops in circuit breaker tests
- Verify async operations complete properly
- Reduce test data size for faster execution

### Flaky tests
- Ensure proper mocking of time-dependent operations
- Use fixed random seeds for reproducibility
- Avoid race conditions in concurrent tests

### Resource leaks
- Verify all file handles are closed
- Check temp directory cleanup
- Monitor memory usage in long-running tests