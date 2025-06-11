# 🔧 AGENT 1: Critical Test Fixes

## PRIMARY MISSION
Fix all test failures caused by Firebase → DynamoDB migration

## CRITICAL FIXES NEEDED

### 1. Fix test_analysis_pipeline_sleep.py
```python
# Line 335 & 356: Replace
from clarity.storage.firestore_client import FirestoreClient
# WITH:
from clarity.services.dynamodb_service import DynamoDBHealthDataRepository
```

### 2. Search & Replace Pattern
```bash
# Find all FirestoreClient references
grep -r "FirestoreClient" tests/ --include="*.py"

# Replace with DynamoDBHealthDataRepository
# Also update:
# - Mock objects
# - Test fixtures
# - Import statements
```

### 3. Common Replacements
- `FirestoreClient()` → `DynamoDBHealthDataRepository()`
- `@patch('clarity.storage.firestore_client.FirestoreClient')` → `@patch('clarity.services.dynamodb_service.DynamoDBHealthDataRepository')`
- Firebase auth mocks → Cognito auth mocks

## COMMANDS TO RUN
```bash
# 1. See all test failures
pytest tests/ml/test_analysis_pipeline_sleep.py -v

# 2. Fix imports
ruff check tests/ --fix

# 3. Run tests again
make test-ml
```

## SUCCESS CRITERIA
- All ML tests pass
- No FirestoreClient references remain
- Tests use DynamoDB mocks instead of Firestore