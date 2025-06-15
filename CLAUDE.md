# CLAUDE.md - CRITICAL PROJECT INFORMATION

## 🚨 CRITICAL: Docker Build Platform Requirements

**ALWAYS BUILD DOCKER IMAGES FOR linux/amd64 PLATFORM FOR AWS ECS DEPLOYMENT**

```bash
# ✅ CORRECT - Always use this:
docker build --platform linux/amd64 -t image-name .

# ❌ WRONG - Never use these:
docker build -t image-name .  # Uses host platform (arm64 on Mac)
docker buildx build --load -t image-name .  # May default to wrong platform
```

## Why This Matters

AWS ECS Fargate ONLY supports `linux/amd64` platform. Building for any other platform (like `arm64` on Apple Silicon Macs) will result in:
- Error: `CannotPullContainerError: image Manifest does not contain descriptor matching platform 'linux/amd64'`
- Task fails to start
- Service shows 0 running tasks
- Application is DOWN

## Active AWS Resources

### ECS Cluster
- **Active Cluster**: `clarity-backend-cluster` (NOT `clarity-cluster` which is INACTIVE)
- **Service**: `clarity-backend-service`
- **Task Definition**: `clarity-backend`

### Load Balancer
- **ALB URL**: http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com
- **Target Group**: `clarity-targets`
- **ACM Certificate ARN**: `arn:aws:acm:us-east-1:124355672559:certificate/183ffae7-82d7-4259-a773-f52bb05c46d8` ✅ ISSUED
- **Domains**: clarity.novamindnyc.com, novamindnyc.com

### ECR Repository
- **Repository**: `124355672559.dkr.ecr.us-east-1.amazonaws.com/clarity-backend`

### ML Models S3 Bucket
- **Bucket**: `clarity-ml-models-124355672559`
- **PAT Model Checksums**:
  - PAT-S: `df8d9f0f66bab088d2d4870cb2df4342745940c732d008cd3d74687be4ee99be`
  - PAT-M: `855e482b79707bf1b71a27c7a6a07691b49df69e40b08f54b33d178680f04ba7`
  - PAT-L: `e8ebef52e34a6f1ea92bbe3f752afcd1ae427b9efbe0323856e873f12c989521`

## Deployment Process

**USE THE CANONICAL DEPLOYMENT SCRIPT**:
```bash
cd ops/
./deploy.sh          # Deploy existing task definition
./deploy.sh --build  # Build new image and deploy
```

The script handles:
1. Building for linux/amd64 platform (CRITICAL!)
2. Tagging and pushing to ECR
3. Updating task definition
4. Deploying to ECS
5. Waiting for stability
6. Health verification

**Current Production Image**: `production-final` (also tagged as `latest`)

## Authentication Fix Status

The authentication 500 error has been fixed in the code:
- `InvalidCredentialsError` is properly raised for wrong credentials (returns 401)
- Exception classes have proper error codes
- Code is correct in `src/clarity/auth/aws_cognito_provider.py`

## Test Suite Status

Current test failures that need fixing:
1. Async/coroutine mock issues in health data tests
2. Mock context manager issues in DynamoDB tests
3. AWS region configuration test
4. Decimal serialization test precision
5. PAT model path security test expectations
6. Integration tests need to handle service unavailable (503)
7. Email verification test expectation

## Important Commands

### Check Service Status
```bash
aws ecs describe-services --cluster clarity-backend-cluster --services clarity-backend-service --region us-east-1
```

### Check Latest Task Failure
```bash
# Get latest task ARN
TASK_ARN=$(aws ecs list-tasks --cluster clarity-backend-cluster --service-name clarity-backend-service --region us-east-1 --query 'taskArns[0]' --output text)

# Describe task to see failure reason
aws ecs describe-tasks --cluster clarity-backend-cluster --tasks $TASK_ARN --region us-east-1
```

### Run Tests Locally
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/clarity --cov-report=term-missing

# Run specific test file
pytest tests/services/test_health_data_service.py -v
```

### Lint and Type Check
```bash
# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Common Pitfalls to Avoid

1. **Never** forget the `--platform linux/amd64` flag when building Docker images
2. **Always** verify the correct ECS cluster name (`clarity-backend-cluster`)
3. **Always** run tests before deploying
4. **Never** deploy without checking the task failure reasons if deployment fails
5. **Always** update this file with new critical information

## Last Updated

- Date: June 15, 2025
- By: Claude (AI Assistant)
- Reason: Successfully deployed to production, cleaned up old resources, and consolidated deployment process