# Pre-Deployment Checklist for Clarity Backend

## 🚨 CRITICAL REMINDERS - READ THIS FIRST!

### Things That FUCKED US UP Before:
1. **WRONG DOCKER PLATFORM**: Building for arm64 instead of linux/amd64
2. **MISSING IAM PERMISSIONS**: Task role needs DynamoDB, S3, and Cognito access
3. **INSUFFICIENT MEMORY**: Task needs 2GB RAM minimum (not 1GB)
4. **WRONG TASK DEFINITION**: Missing environment variables
5. **WRONG ECS CLUSTER**: Use `clarity-backend-cluster` NOT `clarity-cluster`

## 🔍 Verification Steps Before Deployment

### 1. Local Testing (MANDATORY)

```bash
# Run all unit tests
pytest

# Run specific auth tests
pytest tests/api/v1/test_auth_api_comprehensive.py -v

# Check test coverage
pytest --cov=src/clarity --cov-report=term-missing
```

### 2. Local Docker Testing (CRITICAL)

```bash
# Build the Docker image for linux/amd64 (ALWAYS use this platform)
docker build --platform linux/amd64 -t clarity-backend:test .

# Run the container locally
docker run --rm -d \
  -p 8000:80 \
  -e AWS_REGION=us-east-1 \
  -e COGNITO_USER_POOL_ID=us-east-1_efXaR5EcP \
  -e COGNITO_CLIENT_ID=7sm7ckrkovg78b03n1595euc71 \
  -e ENVIRONMENT=development \
  --name clarity-test \
  clarity-backend:test

# Wait for container to start
sleep 5

# Run smoke tests against local container
./scripts/smoke-test.sh http://localhost:8000

# Check container logs if tests fail
docker logs clarity-test

# Stop test container
docker stop clarity-test
```

### 3. Auth Endpoint Smoke Tests

The smoke test script (`scripts/smoke-test-auth.sh`) verifies:

| Test | Expected Status | Description |
|------|----------------|-------------|
| Health Check | 200 | API is running |
| Register - Valid Password | 201 | User created successfully |
| Register - Weak Password | 400 | Password policy violation |
| Register - Duplicate Email | 409 | User already exists |
| Login - Valid Credentials | 200 | Returns auth tokens |
| Login - Wrong Password | 401 | Invalid credentials |
| Login - Non-existent User | 401 | User not found |
| Invalid JSON | 422 | Validation error |

### 4. Manual Verification Commands

```bash
# Test registration with valid password (uppercase + lowercase + numbers)
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","display_name":"Test User","password":"SecurePass123"}'

# Test registration with weak password (should return 400, not 500)
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test2@example.com","display_name":"Test User","password":"weakpassword"}'
```

### 5. Deployment Process

```bash
# Deploy with new build
cd ops/
./deploy.sh --build

# Deploy existing image
cd ops/
./deploy.sh
```

### 6. Post-Deployment Verification

```bash
# Run smoke tests against production
./scripts/smoke-test.sh https://clarity.novamindnyc.com

# Check ECS service status
aws ecs describe-services \
  --cluster clarity-backend-cluster \
  --services clarity-backend-service \
  --region us-east-1

# Check recent logs
aws logs tail /ecs/clarity-backend --since 5m --region us-east-1
```

## ⚠️ Common Issues to Avoid

1. **Wrong Docker Platform**: ALWAYS use `--platform linux/amd64`
   - AWS ECS Fargate ONLY supports linux/amd64
   - Error: `CannotPullContainerError: image Manifest does not contain descriptor matching platform`
2. **Missing Environment Variables**: Ensure ALL env vars are in task definition
   - Check with: `aws ecs describe-task-definition --task-definition clarity-backend --region us-east-1`
3. **Password Policy**: Cognito requires uppercase + lowercase + numbers
4. **500 vs 400 Errors**: Password validation should return 400, not 500
5. **IAM Permissions**: Task role MUST have:
   - `AmazonDynamoDBFullAccess`
   - `AmazonS3ReadOnlyAccess` (for ML models)
   - `AmazonCognitoPowerUser`
6. **Memory Issues**: Task MUST have 2048 MB memory (not 1024)
   - OOM kills will cause continuous task restarts

## 🚨 Rollback Procedure

If deployment fails:

```bash
# Get previous task definition revision
PREV_REVISION=$(aws ecs describe-service \
  --cluster clarity-backend-cluster \
  --service clarity-backend-service \
  --region us-east-1 \
  --query 'service.taskDefinition' \
  --output text | sed 's/.*://' | awk '{print $1-1}')

# Rollback to previous version
aws ecs update-service \
  --cluster clarity-backend-cluster \
  --service clarity-backend-service \
  --task-definition clarity-backend:$PREV_REVISION \
  --force-new-deployment \
  --region us-east-1
```

## 🔐 Current Production Configuration

### ECS Resources:
- **Cluster**: `clarity-backend-cluster` ✅ ACTIVE
- **Service**: `clarity-backend-service`
- **Task Definition**: `clarity-backend:51` (2GB RAM)
- **Task Role**: `arn:aws:iam::124355672559:role/clarity-backend-task-role`

### Cognito Resources:
- **User Pool ID**: `us-east-1_efXaR5EcP`
- **Client ID**: `7sm7ckrkovg78b03n1595euc71`

### S3 ML Models:
- **Bucket**: `clarity-ml-models-124355672559`
- **Models**: PAT-S, PAT-M, PAT-L with checksums

## ✅ Deployment Success Criteria

- [ ] All unit tests pass
- [ ] Docker image builds successfully for linux/amd64
- [ ] Local smoke tests pass (all endpoints return expected status codes)
- [ ] Container runs without errors locally
- [ ] Deployment completes without errors
- [ ] Production smoke tests pass
- [ ] No 500 errors in CloudWatch logs
- [ ] Frontend can successfully register/login users
- [ ] Task has 2GB memory allocated
- [ ] Task role has all required permissions
- [ ] All environment variables are present

## 🛠️ Validation Tools

```bash
# Run this BEFORE deploying to catch configuration issues
cd ops/
./validate-config.sh
```

---

Last Updated: June 15, 2025 (After deployment hell)