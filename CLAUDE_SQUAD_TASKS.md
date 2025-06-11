# Claude Squad Parallel Tasks

## Agent 1: Fix Linting Errors
```bash
# Fix auto-fixable linting errors
ruff check . --fix

# Focus on critical source code issues:
- Remove FirestoreClient references in tests/ml/test_analysis_pipeline_sleep.py
- Fix undefined name errors (lines 335, 356)
- Fix import errors in dynamodb_client.py
```

## Agent 2: Fix Test Failures
```bash
# Run tests and fix failures
make test

# Priority fixes:
- Update test_analysis_pipeline_sleep.py to use DynamoDB instead of Firestore
- Fix any auth test failures related to AWS Cognito
- Update mocked dependencies
```

## Agent 3: Clean AWS Resources
```bash
# Delete old ECR images
aws ecr batch-delete-image \
  --repository-name clarity-backend \
  --image-ids file://ecr-images-to-delete.json

# List old task definitions to clean
aws ecs list-task-definitions --family-prefix clarity --status INACTIVE

# Document current running resources for cost tracking
```

## Agent 4: Update TaskMaster Status
```bash
# Update completed tasks
tm set-status 3 done  # Cognito migration complete
tm set-status 4 done  # DynamoDB migration complete  
tm set-status 6 done  # All 60 endpoints working

# Check remaining tasks
tm next
```

## Coordination Notes:
- All agents should work in the same branch: nuclear-cleanup-deployment
- Commit frequently with clear messages
- Share findings in this file