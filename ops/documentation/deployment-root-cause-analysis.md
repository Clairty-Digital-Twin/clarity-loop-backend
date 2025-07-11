# Root Cause Analysis - ECS Deployment Failures

## Investigation Results

### 1. ALB Health Check Configuration ✅ NOT THE ISSUE
- Health check path: `/health` (correct)
- Interval: 30s, Timeout: 10s (reasonable)
- Currently has 1 healthy target (172.31.32.70)
- ALB configuration is correct

### 2. Task Role Permissions 🔴 ROOT CAUSE FOUND
**Critical Finding**: Task role only had `AmazonS3ReadOnlyAccess` but application needs:
- ❌ DynamoDB access (4 tables)
- ❌ Secrets Manager access (for GEMINI_API_KEY)
- ❌ SQS/SNS access
- ❌ Cognito admin access
- ✅ S3 access (but read-only)

**Actions Taken**:
1. Created comprehensive IAM policy with all required permissions
2. Attached policy to `clarity-backend-task-role`
3. Added SecretsManagerReadWrite to execution role

### 3. Docker Image Status ✅ VERIFIED
- Latest image (352303a7) is in ECR
- Tagged as both commit SHA and `latest`
- Ready for deployment

### 4. Current Service State
- 1 running task with old image (task def 140)
- Task is healthy and serving traffic
- Multiple stuck deployments need cleanup

## Root Cause Summary

The deployment failures were caused by **insufficient IAM permissions**. When tasks started:
1. They couldn't read the GEMINI_API_KEY from Secrets Manager
2. They couldn't access DynamoDB tables
3. Container would crash on startup
4. ECS would retry, fail, and timeout after 10 minutes

## Fix Implementation

### Immediate Actions
1. ✅ Added comprehensive IAM policy to task role
2. ✅ Added Secrets Manager access to execution role
3. 🔄 Deploy latest code with monitoring

### Deployment Strategy
1. Stop all stuck deployments
2. Create new task definition with latest image
3. Deploy with circuit breaker enabled
4. Monitor CloudWatch logs during deployment

## Verification Steps
1. Task starts successfully
2. No permission errors in CloudWatch logs
3. Health check passes
4. Service reaches steady state
5. Smoke tests pass